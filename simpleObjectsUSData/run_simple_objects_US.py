# This script runs a k-Wave US simulation script from the command line.

# %% 
# import necessary libraries
import os
import sys
import json
import pytz
import time
import mat73
import psutil
import platform
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import scipy.io as sio
from functools import partial
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from multiprocessing import Pool, cpu_count
from openpyxl.styles import Font, Alignment
from openpyxl import Workbook, load_workbook
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RegularGridInterpolator as rgi
# load the customPyFunctions.py file to load functions from it
# it is located in the same folder as this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from customPyFunctions import norm_to_one, gaussian, fourier, realFourier, logCompression


# %%
###########################
# Custom defined functions
###########################

##################################################################
# Define functions needed to batch creation of images
##################################################################
# Function run by worker processes during local batch creation
def generate_frame_local(combination_and_params, original_scan_lines, distance_x, distance_y, time_array, t0, ds, medium_y_starting_index, number_scan_lines, active_transducer_width, central_frequency, aws_save_location, aws_file_name):
    index, combination, base_params = combination_and_params
    this_US_scan_params = base_params.copy()
    this_US_scan_params.update(combination)
    aws_file_name = aws_file_name.replace('.png', f'{index+1}.png')
    aws_batch_upload(this_US_scan_params, 
                     original_scan_lines=original_scan_lines, 
                     distance_x=distance_x, 
                     distance_y=distance_y, 
                     time_array=time_array, 
                     t0=t0, 
                     ds=ds, 
                     medium_y_starting_index=medium_y_starting_index, 
                     number_scan_lines=number_scan_lines, 
                     active_transducer_width=active_transducer_width, 
                     central_frequency=central_frequency, 
                     mkv_flag=2, 
                     aws_save_location=aws_save_location, 
                     aws_file_name=aws_file_name)
    return index, this_US_scan_params


##################################################################
# Define functions to take inputs from an Excel file
##################################################################
def close_excel_file():
    """
    Closes all running Excel processes to prevent permission issues 
    when saving or modifying the Excel file.
    """
    for process in psutil.process_iter(attrs=['pid', 'name']):
        if process.info['name'] and 'EXCEL' in process.info['name'].upper():
            print("\nClosing Excel to prevent permission issues...")
            process.kill()
            time.sleep(2)  # Allow time for Excel to close

def wait_for_file_save(file_path, initial_mod_time):
    """
    Waits until the specified Excel file is modified (saved) before continuing.

    Args:
        file_path (str): Path to the Excel file being monitored.
        initial_mod_time (float): Initial modification time of the file.
    """
    print("\nPlease enter values in the Excel file. Save when done.")
    
    while True:
        time.sleep(2)  # Check every 2 seconds
        try:
            current_mod_time = os.path.getmtime(file_path)  # Get last modified time
            if current_mod_time != initial_mod_time:  # Check if file was saved
                print("\nExcel file saved. Continuing execution...")
                break
        except PermissionError:
            pass  # If file is locked by Excel, keep waiting


##################################################################
# Define a function to evaluate the US scan image
##################################################################
def eval_US_scan(params_dict, original_scan_lines, time_array, t0, central_frequency, gain_flag=True):
    # Prepare a copy of the original_scan_lines and time_array
    new_scan_lines = original_scan_lines.copy()
    scan_line_time_array = time_array.copy()

    # Remove input signal from the new_scan_lines
    # calculate the number of points to remove from the new_scan_lines
    n_points_remove = np.round(params_dict['input_signal_max_extent']/np.mean(np.diff(scan_line_time_array))).astype(int)
    # remove the input signal from the new_scan_lines
    new_scan_lines[:, :n_points_remove] = 0

    # Perform Time Gain Compensation (TGC) on the new_scan_lines.
    # Create a radius variable assuming that t0 corresponds to the 
    # middle of the input signal.
    r = params_dict['c0'] * (scan_line_time_array - t0 ) / 2; # in [m]
    # define absorption value and convert to correct units
    tgc_alpha_db_cm = params_dict['alpha_coeff'] * (central_frequency * 1e-6)**params_dict['alpha_power']; # in [dB/(cm*MHz^params_dict['alpha_coeff'])]
    tgc_alpha_np_m = tgc_alpha_db_cm / 8.686 * 100; # in [Np]
    # evaluate time gain compensation based on attenuation value 
    # and round trip distance
    tgc = np.exp(tgc_alpha_np_m * 2 * r)
    # apply the time gain compensation to each of the scan lines
    new_scan_lines = new_scan_lines * tgc

    # Perform frequency filtering on the new_scan_lines
    # get the frequency axis 
    freq = np.fft.rfftfreq(len(scan_line_time_array), d=np.mean(np.diff(scan_line_time_array)))
    # filter the scan lines using filter frequency and FWHM bandwidth 
    frequency_filter = gaussian(freq, 1, params_dict['filter_frequency'], params_dict['filter_bandwidth']/np.sqrt(8*np.log(2)))
    # initialize filtered scan lines
    new_scan_lines_filtered = np.zeros_like(new_scan_lines)
    # run a loop over all scan lines to filter them
    for i in range(new_scan_lines.shape[0]):
        # get the FFT of the scan line
        _, input_scan_line_fft = realFourier(scan_line_time_array, new_scan_lines[i, :])
        # filter the scan line using the frequency filter
        new_scan_lines_filtered[i, :] = np.fft.irfft(input_scan_line_fft * frequency_filter, norm='ortho', n=len(scan_line_time_array))

    # Perform envelope detection on the new_scan_lines
    # loop over all scan lines 
    for i in range(new_scan_lines_filtered.shape[0]):
        # get the envelope of the scan line
        new_scan_lines_filtered[i, :] = np.abs(hilbert(new_scan_lines_filtered[i, :]))
    # Perform normalised log compression on the new_scan_lines
    new_scan_lines_filtered = logCompression(new_scan_lines_filtered, params_dict['compression_ratio'], True)

    # Perform scan conversion, i.e., upsample the new_scan_lines
    if params_dict['scale_factor'] == 1:
        # no change
        new_scan_lines_filtered_upsampled = new_scan_lines_filtered
    else:
        # define arrays depending on the size of new_scan_lines
        indices_initial = [np.arange(0, new_scan_lines_filtered.shape[0]), np.arange(0, new_scan_lines_filtered.shape[1])]
        # get the meshgrid
        mesh_initial = np.meshgrid(indices_initial[0], indices_initial[1], indexing='ij', sparse=True)
        # perform 2D interpolation using the meshgrid
        interp_new_scan_lines = rgi((np.squeeze(indices_initial[0]), np.squeeze(indices_initial[1])), new_scan_lines_filtered, bounds_error=False, method='linear')
        # define the final indices
        indices_final = [np.arange(0, new_scan_lines_filtered.shape[0], 1/params_dict['scale_factor'])[:-params_dict['scale_factor']+1], np.arange(new_scan_lines_filtered.shape[1])]
        # get the final points 
        points_final = []
        for i in range(len(indices_final[0])):
            for j in range(len(indices_final[1])):
                points_final.append([indices_final[0][i], indices_final[1][j]])
        # get the interpolated new_scan_lines
        new_scan_lines_filtered_upsampled = interp_new_scan_lines(points_final).reshape(-1, new_scan_lines_filtered.shape[1])
        # replace the nan in both the upsampled new_scan_lines if any
        new_scan_lines_filtered_upsampled[np.isnan(new_scan_lines_filtered_upsampled)] = new_scan_lines_filtered.min()

    if gain_flag:
        # Apply gain 
        new_scan_lines_filtered_upsampled *= 10**(params_dict['gain']/20)
        # Apply contrast thresholds 
        new_scan_lines_filtered_upsampled = np.clip(new_scan_lines_filtered_upsampled, 
                                                    10 ** (params_dict['contrast_min'] / 20), 
                                                    10 ** (params_dict['contrast_max'] / 20))
        # Perform TGC adjustment on the new_scan_lines_filtered_upsampled
        for tgc_value in range(8):
            # Apply TGC to the respective region
            # first get the start and end indices of the corresponding region
            tgc_start_index = np.round(params_dict['input_signal_max_extent']/np.mean(np.diff(scan_line_time_array))).astype(int)
            region_length = (new_scan_lines_filtered_upsampled.shape[1] - tgc_start_index) // 8
            start_idx = tgc_start_index + tgc_value * region_length
            end_idx = tgc_start_index + (tgc_value + 1) * region_length if tgc_value < 7 else new_scan_lines_filtered_upsampled.shape[1]
            new_scan_lines_filtered_upsampled[:, start_idx:end_idx] *= 10**(params_dict['TGC_{}'.format(tgc_value + 1)] / 20)

    return new_scan_lines_filtered_upsampled           


##################################################################
# Define a function that prepares a new image for AWS upload for 
# the particular combination of scan parameters provided
##################################################################
def aws_batch_upload(aws_US_scan_params, 
                     original_scan_lines, 
                     distance_x, 
                     distance_y, 
                     time_array, 
                     t0, 
                     ds, 
                     medium_y_starting_index, 
                     number_scan_lines, 
                     active_transducer_width, 
                     central_frequency, 
                     mkv_flag=0, 
                     aws_save_location=None, 
                     aws_file_name=None):
    # Get the scan lines for this particular set of scan parameters
    aws_scan_lines = eval_US_scan(aws_US_scan_params, original_scan_lines, time_array, t0, central_frequency)
    # Normalize and scale scan lines to 0-255
    aws_scan_lines = norm_to_one(aws_scan_lines - aws_scan_lines.min()) * 255

    # Return the image matrix or save it
    if mkv_flag == 1:
        return aws_scan_lines

    elif mkv_flag == 2:
        # Save the figure
        # Get axes and positions
        temp_t0_distance, temp_x_axis_length, _, temp_transducer_positions = get_axes_and_positions(aws_US_scan_params, aws_scan_lines.shape, time_array, t0, ds, medium_y_starting_index, number_scan_lines, active_transducer_width)

        # Plot and save the figure
        fig_aws, ax_aws = plt.subplots(figsize=(9, 6))
        ax_aws.imshow(aws_scan_lines, origin='lower', cmap='gray', extent=[-temp_t0_distance * 1e3, 
                                                                            temp_x_axis_length - temp_t0_distance * 1e3, 
                                                                            temp_transducer_positions[0], 
                                                                            temp_transducer_positions[-1]])
        ax_aws.set_xlim([distance_x[0], distance_x[-1]])
        ax_aws.set_ylim([-0.5 * distance_y.max(), 0.5 * distance_y.max()])                
        # turn off the xticks and yticks
        ax_aws.set_xticks([])
        ax_aws.set_yticks([])

        # Save the figure
        fig_aws.savefig(os.path.join(aws_save_location, aws_file_name), dpi=300, bbox_inches='tight')
        plt.close(fig_aws)

    else:
        # Save the figure and parameters        
        # Get axes and positions
        temp_t0_distance, temp_x_axis_length, _, temp_transducer_positions = get_axes_and_positions(aws_US_scan_params, aws_scan_lines.shape, time_array, t0, ds, medium_y_starting_index, number_scan_lines, active_transducer_width)

        # Plot and save the figure
        fig_aws, ax_aws = plt.subplots(figsize=(9, 6))
        ax_aws.imshow(aws_scan_lines, origin='lower', cmap='gray', extent=[-temp_t0_distance * 1e3, 
                                                                            temp_x_axis_length - temp_t0_distance * 1e3, 
                                                                            temp_transducer_positions[0], 
                                                                            temp_transducer_positions[-1]])
        ax_aws.set_title('US scan')
        ax_aws.set_xlabel('x-axis distance (in mm)')
        ax_aws.set_ylabel('y-axis distance (in mm)')
        ax_aws.set_xlim([distance_x[0], distance_x[-1]])
        ax_aws.set_ylim([-0.5 * distance_y.max(), 0.5 * distance_y.max()])                

        # Ensure save location exists
        if not os.path.exists(aws_save_location):
            os.makedirs(aws_save_location)
        # Save the figure
        fig_aws.savefig(os.path.join(aws_save_location, aws_file_name), dpi=300, bbox_inches='tight')
        plt.close(fig_aws)

        # Save parameters as JSON 
        aws_US_scan_params_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in aws_US_scan_params.items()}
        with open(os.path.join(aws_save_location,  aws_file_name.replace('.png', '_metadata.json')), 'w') as params_file:
            json.dump(aws_US_scan_params_serializable, params_file, indent=4)


##################################################################    
# Define a function to get axes lengths and transducer positions 
# depending on scan parameters
##################################################################
def get_axes_and_positions(US_scan_params, 
                           scan_lines_filtered_upsampled_shape, 
                           time_array, 
                           t0, 
                           ds, 
                           medium_y_starting_index, 
                           number_scan_lines, 
                           active_transducer_width, ):
    # assuming t0 as the start of the input signal, 
    # convert it into a unit of distance
    t0_distance = US_scan_params['c0'] * t0 / 2; # in [m]

    # create a radius variable assuming that t0 
    # corresponds to the middle of the input signal
    r = US_scan_params['c0'] * time_array / 2 - t0_distance; # in [m]

    # get the lengths in mm for scan_lines_filtered_upsampled
    x_axis_length = scan_lines_filtered_upsampled_shape[1] * 0.5e3 * US_scan_params['c0'] * np.mean(np.diff(time_array))

    # get all the positions where transducer elements were placed for the imaging
    # note that medium_y_starting_index was assigned according to MATLAB indexing
    if US_scan_params['scale_factor'] == 1:
        transducer_positions = np.arange(medium_y_starting_index - 1, 
                                        medium_y_starting_index - 1 + number_scan_lines * active_transducer_width, 
                                        active_transducer_width) + (active_transducer_width - 1)/2
    else:
        transducer_positions = np.arange(medium_y_starting_index - 1, 
                                        medium_y_starting_index - 1 + number_scan_lines * active_transducer_width, 
                                        active_transducer_width/US_scan_params['scale_factor'])[:-US_scan_params['scale_factor']+1] + ((active_transducer_width/US_scan_params['scale_factor']) - 1)/2
    # convert to mm and subtract the mean to center the transducer positions
    transducer_positions = transducer_positions * ds * 1e3 
    transducer_positions = transducer_positions - np.mean(transducer_positions)

    return t0_distance, x_axis_length, r, transducer_positions


# %%
if __name__ == "__main__":
    # give the name of folder containing US scan lines
    current_folder = '<ENTER_CURRENT_FOLDER_PATH_HERE>'

    # Create folders for runtime and to save final simulation results
    # get the location of the parent folder of this python script
    parent_folder = os.path.dirname(os.path.abspath(__file__))
    # extract 'sim_name', 'transducer_active_elements', and sweep_flag from the current_folder
    sim_name = os.path.basename(os.path.dirname(os.path.dirname(current_folder)))
    # for sweep_flag, put it as 0 if there is 'sweep' in current_folder name
    # or 0 if 'single_acq' is in current_folder name
    sweep_flag = 0 if 'sweep' in current_folder else 1
    # print them
    print(f'Simulation name: {sim_name}')
    print(f'Sweep flag: {sweep_flag}')


    # %%
    ##################################################################
    # Define the inputs for the MATLAB script
    ##################################################################
    # give the PML size in grid points
    pml_size = 20

    # give the folder location where data files are to be saved
    data_path = current_folder

    # give the location to save I/O save files during k-Wave run
    # create a folder called 'io_files' if it doesn't exist already
    io_path = os.path.join(current_folder, 'io_files')
    if not os.path.exists(io_path):
        os.makedirs(io_path)

    # give the isotropic grid spacing
    ds = 0.0635e-3 # in [m]

    # give the cfl number
    cfl = 0.1

    # give the transducer properties
    # get a realistic number for the total number of elements
    transducer_total_elements = 80
    # set the number of active elements depending on sweep_flag
    if sweep_flag == 0:
        transducer_active_elements = 1 
    else:
        transducer_active_elements = transducer_total_elements
    transducer_element_width = 4 # in [grid points]
    transducer_element_length = 189 # in [grid points]
    transducer_element_spacing = 0 # in [grid points]
    transducer_radius = np.inf # in [m]
    transducer_focus_distance = 65e-3 # in [m]
    transducer_elevation_focus_distance = 65e-3 # in [m]
    transducer_steering_angle = 0 # in [degrees]
    transducer_transmit_apodization = 'Rectangular' # 'Hanning' or 'Rectangular'
    transducer_receive_apodization = 'Rectangular' # 'Hanning' or 'Rectangular'

    # give the US input signal
    global central_frequency
    central_frequency = 2.7e6 # Hz
    fwhm_bandwidth = 85 # as a percentage of the central frequency
    global t0
    t0 = 1e-6 # in [s]

    # calculate the width of the active transducer in grid points
    active_transducer_width = transducer_active_elements * transducer_element_width + \
                            (transducer_active_elements - 1) * transducer_element_spacing

    # give Nx, Ny, Nz as the size of the simulation grid without PML
    Nx = 256 - 2*pml_size # in grid points
    # Ny = (2**np.ceil(np.log(active_transducer_width + 20 + 2*pml_size)/np.log(2))-2*pml_size).astype(int) # in grid points
    Ny = active_transducer_width # in grid points
    Nz = 256 - 2*pml_size # in grid points

    # give the active transducer position in the grid
    # according to MATLAB indexing
    active_transducer_x_axis_position = 1 # in [grid points]
    active_transducer_y_axis_position = 1 + Ny//2 - active_transducer_width//2 # in [grid points]
    active_transducer_z_axis_position = 1 + Nz//2 - transducer_element_length//2 # in [grid points]

    # get the dimensions of the object
    grid_size = [Nx, Nx, Nx] # in [grid points]
    grid_spacing = [ds*1e3, ds*1e3, ds*1e3] # in [mm]

    # set the number of scan lines, and medium_y_starting_index
    # 0 = sweep, 1 = single acquisition
    if sweep_flag == 0:
        # Set the medium_y_starting_index.
        medium_y_starting_index = 1 # in [grid points]

        # Set the number of scan lines
        number_scan_lines = int((grid_size[1]*grid_spacing[1]*1e-3//ds)//transducer_element_width)

    else:
        # Set the number of scan lines to 1.
        number_scan_lines = 1
        # Set the medium_y_starting_index depending on the size of the object
        # relative to the transducer
        if grid_size[1] * grid_spacing[1] < Ny * ds * 1e3:
            medium_y_starting_index = 1
            # change the Ny to the size of the object
            Ny = int(grid_size[1] * grid_spacing[1] * 1e-3 // ds)
            # now change the total number of transducer elements
            transducer_total_elements = np.floor(Ny//transducer_element_width).astype(int)
            # now change the number of active elements
            transducer_active_elements = transducer_total_elements
            # now change the active transducer width
            active_transducer_width = transducer_active_elements * transducer_element_width + \
                                    (transducer_active_elements - 1) * transducer_element_spacing
            # now change the active transducer y-axis position
            active_transducer_y_axis_position = Ny//2 - active_transducer_width//2 # in [grid points]

        else:
            medium_y_starting_index = int(np.round(grid_size[1]*grid_spacing[1]*1e-3//(2*ds))) - Ny//2 # in [grid points]


    # %%
    ##################################################################
    # Perform post-processing on simulation results
    ##################################################################
    # Load simulation specific .mat files saved during runtime
    data = mat73.loadmat(os.path.join(current_folder, 'simulation_specifics.mat'))
    global time_array
    time_array = np.squeeze(np.array(data['time_array']))
    input_signal = norm_to_one(np.squeeze(np.array(data['input_signal'])))
    c0 = data['c0']

    # Load individual scan lines recorded at every medium position 
    # by the active elements of the transducer.
    # Look for the substring "medium_position=" in the current_folder
    scan_lines_files = [f for f in os.listdir(current_folder) if 'medium_position=' in f]
    # Arrange the files in ascending order of the value after the 
    # substring and before .mat
    scan_lines_files.sort(key=lambda x: int(x.split('medium_position=')[1].split('.mat')[0]))

    # load the scan_lines from the individual files
    scan_lines = []
    for file in scan_lines_files:
        scan_lines.append(mat73.loadmat(os.path.join(current_folder, file))['scan_line'])
    # convert to a numpy array
    scan_lines = np.squeeze(np.array(scan_lines))
    # reshape scan_lines properly
    scan_lines = scan_lines.reshape(-1, len(np.squeeze(time_array)))
    # print the shape of scan_lines
    print(f'Scan_lines shape: {scan_lines.shape}')
    
    # if sweep_flag == 1, then replace scan_lines with sensor_data. 
    # This is the case for the 'single_acq' mode.
    if sweep_flag == 1:
        # load sensor_data from each of the scan_lines
        sensor_data = []
        for file in scan_lines_files:
            sensor_data.append(mat73.loadmat(os.path.join(current_folder, file))['sensor_data'])
        sensor_data = np.array(np.squeeze(sensor_data))
        # print the shape of sensor_data
        print(f'sensor_data shape: {sensor_data.shape}')
        # replace scan_lines with sensor_data
        scan_lines = sensor_data
        # reshape scan_lines properly
        scan_lines = scan_lines.reshape(-1, len(np.squeeze(time_array)))

    # create a copy of the original scan_lines
    global original_scan_lines
    original_scan_lines = scan_lines.copy()


    # %%
    ##################################################################
    # Get the image axes dimensions
    ##################################################################
    # get the distances along the x and y directions
    global distance_x
    distance_x = np.arange(grid_size[0]) * grid_spacing[0]
    global distance_y
    distance_y = np.arange(grid_size[1]) * grid_spacing[1]

    ##################################################################
    # Define a dictionary to initialize US scan parameters
    ##################################################################
    US_scan_params = {'object_alpha': 0.5,
                        'scan_line_index': (2*scan_lines.shape[0] - 1)//2, # middle scan line
                        'scale_factor': 2, 
                        'input_signal_max_extent': t0 + 10/((2*np.pi*((fwhm_bandwidth*central_frequency/100)/np.sqrt(8*np.log(2))))), # in [s]
                        'c0': c0,
                        'compression_ratio': 3,
                        'filter_frequency': central_frequency, # in [Hz]
                        'filter_bandwidth': fwhm_bandwidth*central_frequency/100, # in [Hz]
                        'contrast_min': 0,
                        'contrast_max': 0,
                        'gain': 0, # in dB
                        'TGC_1': 0, # Initialize TGC values for 8 regions
                        'TGC_2': 0,
                        'TGC_3': 0,
                        'TGC_4': 0,
                        'TGC_5': 0,
                        'TGC_6': 0,
                        'TGC_7': 0,
                        'TGC_8': 0,
                        'alpha_coeff': 0.75, # in [dB/(cm*MHz^US_scan_params['alpha_coeff'])]
                        'alpha_power': 1.5}
    
    # store the slider names in a list
    slider_names = ['alpha', 
                    'scan line index',
                    'scale factor',
                    'input signal max extent',
                    'c0',
                    'compression ratio',
                    'filter frequency',
                    'filter bandwidth',
                    'contrast min',
                    'contrast max',
                    'gain', 
                    'TGC 1',
                    'TGC 2',
                    'TGC 3',
                    'TGC 4',
                    'TGC 5',
                    'TGC 6',
                    'TGC 7',
                    'TGC 8',
                    'alpha coeff', 
                    'alpha power']        

    # get the US scan lines after processing
    scan_lines_filtered_upsampled = eval_US_scan(US_scan_params, original_scan_lines, time_array, t0, central_frequency, gain_flag=False)

    # change the contrast_min\max values in US_scan_params
    US_scan_params['contrast_min'] = np.round(20*np.log10(scan_lines_filtered_upsampled.min()), 2)
    US_scan_params['contrast_max'] = np.round(20*np.log10(scan_lines_filtered_upsampled.max()), 2)

    # get the axes and positions
    t0_distance, x_axis_length, r, transducer_positions = get_axes_and_positions(US_scan_params, scan_lines_filtered_upsampled.shape, time_array, t0, ds, medium_y_starting_index, number_scan_lines, active_transducer_width)

    # %%
    ##################################################################
    # Create a local batch of images using inputs from an Excel 
    # sheet where ranges are provided for every parameter and each 
    # combination of inputs will be used to create new images.
    ##################################################################    
    # Define the file path for the Excel file
    excel_file_name = 'local_batch_uploads.xlsx'
    excel_file_path = os.path.join(parent_folder, excel_file_name)
    # Give the location of the folder where the images will be saved
    batch_folder_save_location = "<ENTER_DESTINATION_FOLDER_PATH>"

    # For the following keys, take the input in logarithmic scale
    log_inputs = ['compression_ratio']
    
    # Load values from the Excel file
    df_updated = pd.read_excel(excel_file_path, usecols=[0, 1, 2, 3, 4])
    print("Processing the Excel file...")
    # Collect parameter ranges from the updated DataFrame
    param_values = []
    param_keys = []
    # certain parameters require their input to be int
    int_params = ['scan_line_index', 'scale_factor', 'compression_ratio']
    # if these parameters are in the list of keys, convert their values to the nearest int
    for i, row in df_updated.iterrows():
        if row['Number'] > 1:  # Only consider parameters with multiple values
            # take into care if the inputs are in logarithmic scale
            if row['Key'] in log_inputs:
                values = 10**np.linspace(row['Minimum'], row['Maximum'], row['Number'])
            else:
                values = np.linspace(row['Minimum'], row['Maximum'], row['Number'])
            # Check if the parameter is in the int_params list
            if row['Key'] in int_params:
                values = np.round(values).astype(int)
            param_values.append(values)
            param_keys.append(row['Key'])
    # Generate all combinations
    param_combinations = list(product(*param_values))
    # Convert to list of dicts with param_keys
    combinations_dicts = [dict(zip(param_keys, values)) for values in param_combinations]

    # Prepare combinations and base params
    pool_inputs = [(i, comb, US_scan_params) for i, comb in enumerate(combinations_dicts)]

    # Use functools.partial to pass additional arguments to generate_frame_local
    generate_frame_local_partial = partial(generate_frame_local, 
                                            original_scan_lines=original_scan_lines, 
                                            distance_x=distance_x,
                                            distance_y=distance_y,
                                            time_array=time_array, 
                                            t0=t0, 
                                            ds=ds,
                                            medium_y_starting_index=medium_y_starting_index, 
                                            number_scan_lines=number_scan_lines, 
                                            active_transducer_width=active_transducer_width, 
                                            central_frequency=central_frequency, 
                                            aws_save_location=batch_folder_save_location, 
                                            aws_file_name = ".png")

    # Process in parallel
    with Pool(processes=cpu_count()) as pool:
        for (index, frame_params) in pool.imap_unordered(generate_frame_local_partial, pool_inputs):
            pass

    print(f"\nUS images created and saved at: {batch_folder_save_location}")


# %%
print(f"\nSCRIPT FINISHED!\n")
    
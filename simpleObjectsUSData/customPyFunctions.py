# this file will be used to call custom functions that are used in a main script

import os
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt 


# print(__doc__)


def fourier(time, f, **kwargs):
    """
    Plots absolute magnitude of fourier transform of f(1xN or Nx1) and returns the fft
    along with the frequency

    Parameters
    ----------
    time: the time signal upon which 'f' is evaluated
    f: the function whose fourier transform is to be
        evaluated (must be in 1xN or Nx1)

    Optional parameters
    -------------------
    shift_flag: boolean to decide whether to perform
                fftshift in the frequency axis or not
    plot_flag: boolean to decide whether to plot the fft

    Returns
    -------
    freq: the frequency axis
    y: the fourier transform of the signal 'f' for frequencies (positive and negative)
        normalized according to the 'ortho' for the argument 'norm' in np.fft.fft()
        See: https://numpy.org/doc/stable/reference/routines.fft.html#normalization

    USAGE
    -----
    fourier(time, f, plot_flag, shift_flag)
    fourier(time, f, plot_flag)
    fourier(time, f)
    """
    # assign values to the optional parameters
    # check if the optional parameters are provided
    shift_flag = kwargs.get('shift_flag', False)
    plot_flag = kwargs.get('plot_flag', False)

    # get the inputs in the correct format
    time = np.squeeze(np.array(time))
    f = np.squeeze(np.array(f))

    # get the FFT amplitudes
    y = np.fft.fft(f, norm='ortho')
    # get the frequency axis
    # freq = np.array([n/(np.mean(np.diff(time)) * len(f)) for n in range(len(y))])
    freq = np.fft.fftfreq(len(np.squeeze(time)), d=np.mean(np.diff(time)))

    # shift the FFT amplitudes depending on the flag
    if shift_flag:
        y = np.fft.fftshift(y)
        freq = np.fft.fftshift(freq)

    if plot_flag == True:
        plt.figure(), plt.plot(freq, np.abs(y))
        plt.title('FFT')
        plt.xlabel('frequency')
        plt.ylabel('Absolute magnitude of FFT')
        plt.grid()

    return freq, y


def gaussian(x, a, mu, sig):
    """
    Returns the gaussian function of the input x

    Parameters
    ----------
    x: the input to the gaussian function
    a: the amplitude of the gaussian
    mu: the mean of the gaussian
    sig: the standard deviation of the gaussian

    Returns
    -------
    y: the gaussian distribution
    """
    y = a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return y


def logCompression(signal, compression_ratio, normalize=False):
    """
    This function compresses the input signal using a logarithmic compression

    Parameters
    ----------
    signal: the input signal
    compression_ratio: the compression ratio
    normalize: a boolean to decide whether to normalize the output or not

    Returns
    -------
    compressed_signal: the compressed signal
    """
    if normalize:
        mx = np.max(signal)
    else:
        mx = 1.0
    a = compression_ratio
    compressed_signal = mx * (np.log10(1 + a * signal / mx) / np.log10(1 + a))
    return compressed_signal


def norm_sum(arr):
    """
    This function normalizes the input array to have a sum of 1

    Parameters
    ----------
    arr: the input array

    Returns
    -------
    arr_norm: the normalized array
    """
    # squeeze the array to remove singleton dimensions
    arr = np.squeeze(arr)

    # if the sum is 0, then return the matrix as it is
    # otherwise, normalize the array to have a maximum sum of 1
    if np.sum(arr) != 0:
        arr = arr / arr.sum()

    return arr


def norm_to_one(mat):
    """
    This function normalizes the input matrix to have a maximum absolute value of 1.

    Parameters
    ----------
    mat: the input matrix

    Returns
    -------
    mat_norm: the normalized matrix
    """
    # squeeze the matrix to remove singleton dimensions
    mat = np.squeeze(mat)

    # if the maximum absolute value of the matrix is 0, then return the matrix as it is
    # otherwise, normalize the matrix to a maximum absolute value of 1
    if np.abs(mat).max() != 0:
        mat = mat / np.abs(mat).max()

    return mat


def realFourier(*args):
    """
    Plots absolute magnitude of fourier transform of f(1xN or Nx1) and returns the fft
    along only for positive frequencies

    Parameters
    ----------
    time: the time signal upon which 'f' is evaluated
    f: the function whose fourier transform is to be
        evaluated (must be in 1xN or Nx1)
    plot_flag: boolean to decide whether to plot the fft

    Returns
    -------
    freq: the positive frequency axis
    y: the fourier transform of the signal 'f' for positive frequencies
        normalized according to the 'ortho' for the argument 'norm' in np.fft.fft()
        See: https://numpy.org/doc/stable/reference/routines.fft.html#normalization

    USAGE
    -----
    fourier(time, f, plot_flag)
    fourier(time, f)
    """
    # store the arguments in correct parameters
    time = np.squeeze(np.array(args[0]))
    f = np.squeeze(np.array(args[1]))

    # check if the correct flag has been provided for plotting
    if len(args) != 3:
        plot_flag = False
    else:
        plot_flag = args[2]

    # get the FFT amplitudes of positive frequencies
    y = np.fft.rfft(f, norm='ortho')
    ## get the positive frequency axis
    # freq = np.array([n/(np.mean(np.diff(time)) * len(f)) for n in range(len(y))])
    freq = np.fft.rfftfreq(len(np.squeeze(time)), d=np.mean(np.diff(time)))

    if plot_flag == True:
        plt.figure(), plt.plot(freq, np.abs(y))
        plt.title('FFT')
        plt.xlabel('frequency')
        plt.ylabel('Absolute magnitude of FFT')
        plt.grid()

    return freq, y


def run_kWave_script(script_name, kWave_location, current_folder):
    """
    This function runs a k-Wave script from Windows command line and logs the output
    
    Parameters
    ----------
    script_name: the name of the script to be run
    kWave_location: the location of the k-Wave folder
    current_folder: the current folder where the log file will be saved

    Returns
    -------
    None
    """
    # check if there's a GPU or CPU in script_name
    if 'GPU' in script_name:
        print('Running the script on the GPU...')
    else:
        print('Running the script on the CPU...')

    # create a folder called 'runtime_scripts' inside kWave_location if it does not exist already
    runtime_scripts_folder = os.path.join(kWave_location, 'runtime_scripts')
    if not os.path.exists(runtime_scripts_folder):
        os.makedirs(runtime_scripts_folder)

    # copy script_name to runtime_scripts_folder
    script_name_runtime = os.path.join(runtime_scripts_folder, os.path.basename(script_name))
    shutil.copy(script_name, script_name_runtime)

    # Create a log file to save the command line output
    # first create a separate folder called simulation_logs in the current_folder
    # if it does not exist already
    simulation_logs_folder = os.path.join(current_folder, 'simulation_logs')
    if not os.path.exists(simulation_logs_folder):
        os.makedirs(simulation_logs_folder)
    log_file = os.path.join(simulation_logs_folder, os.path.basename(script_name).replace('.m', 'simulation_log.txt'))

    # Command to run the MATLAB script
    command = (f'matlab -batch "disp([\'Current directory: \', pwd]);'
               f'addpath(genpath(\'{kWave_location}\'));'
               f'run(\'{script_name_runtime}\');'
               f'disp(\'FINISHED RUNNING THE MATLAB SCRIPT.\');'
               f'exit;"')

    # now run the copied MATLAB script in the runtime_scripts_folder 
    # after having switched to the kWave_location directory
    os.chdir(kWave_location)

    # Run the MATLAB script using subprocess, 
    # display everything on the command line
    # and also save the output to the log file
    with open(log_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print to command line
            f.write(line)  # Write to log file
            f.flush()  # Ensure it writes to the file immediately
        for line in iter(process.stderr.readline, ''):
            print(line, end='')  # Print to command line
            f.write(line)  # Write to log file
            f.flush()  # Ensure it writes to the file immediately
        process.wait()

    # delete the copy of the MATLAB script from the runtime_scripts_folder
    os.remove(script_name_runtime)      
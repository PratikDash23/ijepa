# this file will be used to create a dataset for IJEPA

import os
import sys
import glob
import pytz
import shutil
import multiprocessing
import subprocess
from datetime import datetime
import re

# define a function to create images for every object in grandparent_names
def create_images_for_objects(object_name, folder_name, test_set, train_set, test_set_folder, train_set_folder, script_path, custom_py_function_path, xlsx_file_path):
    # create a subfolder in train_set_folder or test_set_folder
    if object_name in test_set:
        object_folder = os.path.join(test_set_folder, object_name)
    else:
        object_folder = os.path.join(train_set_folder, object_name)
    os.makedirs(object_folder, exist_ok=True)

    # load the script
    with open(script_path, 'r') as script_file:
        script_content = script_file.read()
    # fix path issues for both Windows and Ubuntu
    if os.name == 'nt':
        # On Windows, use forward slashes for Blender compatibility
        folder_name_fixed = folder_name.replace('\\', '/')
        object_folder_fixed = object_folder.replace('\\', '/')
    else:
        # On Unix-like systems, paths are already with forward slashes
        folder_name_fixed = folder_name
        object_folder_fixed = object_folder
    script_content = script_content.replace("<ENTER_CURRENT_FOLDER_PATH_HERE>", folder_name_fixed)
    script_content = script_content.replace("<ENTER_TRAIN_OR_TEST_FOLDER_PATH>", object_folder_fixed)
    # write the modified script and save it in the object folder
    modified_script_path = os.path.join(object_folder, "run_simple_objects_US.py")
    with open(modified_script_path, 'w') as modified_script_file:
        modified_script_file.write(script_content)

    # copy the script mentioned by customPyFunctionPath to the object folder
    shutil.copy(custom_py_function_path, object_folder)

    # copy the xlsx file to the object folder
    shutil.copy(xlsx_file_path, object_folder)

    # run the script using the python interpreter placed in ai_venv\Scripts\python.exe located in the grandparent folder
    python_executable = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_venv", "Scripts", "python.exe")
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"Python executable {python_executable} does not exist. Please check the path.")    
   
    # Run the command using subprocess for better error handling
    try:
        process = subprocess.Popen(
            [python_executable, modified_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Print stdout and stderr line by line as they are produced
        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")

if __name__ == "__main__":
    # get the parent folder of this script
    parent_folder = os.path.dirname(os.path.abspath(__file__))
    grandparent_folder = os.path.dirname(parent_folder)

    # there's a folder called "scan_data" in the parent folder
    # get the locations till level 5 of subfolders in that folder
    scan_data_folder = os.path.join(parent_folder, "scan_data")
    # get the locations of all subfolders in the scan_data folder
    # Get all subfolders up to 3 levels deep (scan_data/*/*/*)
    subfolders = glob.glob(os.path.join(scan_data_folder, '*', '*', '*'))
    subfolders = [f for f in subfolders if os.path.isdir(f)]
    # for all subfolders, get the names of their grandparents
    grandparent_names = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in subfolders]

    # give the object names
    object_names = ["solid_sphere1", 
                    "solid_sphere2", 
                    "hollow_sphere", 
                    "solid_cylinder", 
                    "hollow_cylinder1",
                    "hollow_cylinder2",
                    "solid_cone",
                    "solid_cuboid",
                    "sphere_with_conical_spikes"]
    
    # arrange the grandparent_names in the same order as object_names
    grandparent_names = [name for name in object_names if name in grandparent_names]

    # create a dictionary called folder_names, with keys as the grandparent names and values as the subfolder paths
    folder_names = {}
    for grandparent_name, subfolder in zip(grandparent_names, subfolders):
        folder_names[grandparent_name] = subfolder

    # select 2 objects for test set from grandparent_names
    test_set = ["hollow_cylinder2", "solid_sphere2"]
    # get the objects for the train set
    train_set = [name for name in grandparent_names if name not in test_set]

    # name the python script to be used later to create the images
    script_path = os.path.join(grandparent_folder, "simpleObjectsUSData", "run_simple_objects_US.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script {script_path} does not exist.")
    
    # get the path of another script called "customPyFunction.py" in the same folder as this script
    custom_py_function_path = os.path.join(parent_folder, "customPyFunctions.py")
    # get the path of an xlsx file called "local_batch_uploads.xlsx" in the same folder as this script
    xlsx_file_path = os.path.join(parent_folder, "local_batch_uploads.xlsx")

    # get the current data and time
    current_time = datetime.now(pytz.timezone("America/Toronto")).strftime('DATE_%Y_%m_%d_TIME_%H_%M_%S')
    # create a folder called "dataset_current_time" in a folder called "dataset" in the grandparent folder
    dataset_folder = os.path.join(grandparent_folder, "dataset", f"dataset_{current_time}")
    os.makedirs(dataset_folder, exist_ok=True)
    # inside this folder, create two subfolders called "train_set" and "test_set"
    train_set_folder = os.path.join(dataset_folder, "train_set")
    test_set_folder = os.path.join(dataset_folder, "test_set")
    os.makedirs(train_set_folder, exist_ok=True)
    os.makedirs(test_set_folder, exist_ok=True)

    for object_name in grandparent_names:
        create_images_for_objects(object_name,
                                  folder_names[object_name],
                                  test_set,
                                  train_set,
                                  test_set_folder,
                                  train_set_folder,
                                  script_path,
                                  custom_py_function_path,
                                  xlsx_file_path)

    # For both train_set and test_set, create a subfolder called "class0"
    # in both these folders, copy the images of all objects in the train_set and test_set respectively
    for set_folder, object_set in [(train_set_folder, train_set), (test_set_folder, test_set)]:
        class0_folder = os.path.join(set_folder, "class0")
        os.makedirs(class0_folder, exist_ok=True)
        for object_name in object_set:
            object_folder = os.path.join(set_folder, object_name)
            # Get the current number of images in the class0 folder to determine the starting index
            existing_images = glob.glob(os.path.join(class0_folder, '*.png'))
            if existing_images:
                # Extract numbers from filenames and find the max
                numbers = []
                for img in existing_images:
                    match = re.search(r'(\d+)', os.path.basename(img))
                    if match:
                        numbers.append(int(match.group(1)))
                start_idx = max(numbers) + 1 if numbers else 1
            else:
                start_idx = 1

            # Copy images from the object folder to the class0 folder with new numbering
            image_files = sorted(glob.glob(os.path.join(object_folder, '*.png')))
            for i, image_file in enumerate(image_files, start=start_idx):
                new_filename = f"{i}.png"
                shutil.copy(image_file, os.path.join(class0_folder, new_filename))

            # delete the object folder after copying the images
            shutil.rmtree(object_folder, ignore_errors=True)
    print("Dataset creation completed successfully.")
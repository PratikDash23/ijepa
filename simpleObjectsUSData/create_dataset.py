# this file will be used to create a dataset for IJEPA

import os
import sys
import glob
import pytz
import shutil
import random
random.seed(42)  # For reproducibility
import multiprocessing
import pandas as pd
import subprocess
from datetime import datetime
import re

# define a function to create images for every object in grandparent_names
def create_images_for_objects(object_name, folder_name, dataset_folder, script_path, custom_py_function_path, xlsx_file_path):
    # create a subfolder in dataset_folder
    object_folder = os.path.join(dataset_folder, object_name)
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
    script_content = script_content.replace("<ENTER_DESTINATION_FOLDER_PATH>", object_folder_fixed)
    # write the modified script and save it in the object folder
    modified_script_path = os.path.join(object_folder, "run_simple_objects_US.py")
    with open(modified_script_path, 'w') as modified_script_file:
        modified_script_file.write(script_content)

    # copy the script mentioned by customPyFunctionPath to the object folder
    shutil.copy(custom_py_function_path, object_folder)

    # copy the xlsx file to the object folder
    shutil.copy(xlsx_file_path, object_folder)

    # run the script using the python interpreter placed in ai_venv\Scripts\python.exe located in the grandparent folder
    if os.name == 'nt':
        python_executable = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ai_venv", "Scripts", "python.exe")
    else:
        python_executable = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_venv", "bin", "python")
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
    scan_data_folder = os.path.join(parent_folder, "scan_data")
    # get the locations of all subfolders in the scan_data folder
    # Get all subfolders up to 3 levels deep (scan_data/*/*/*)
    subfolders = glob.glob(os.path.join(scan_data_folder, '*', '*', '*'))
    subfolders = [f for f in subfolders if os.path.isdir(f)]
    # for all subfolders, get the names of their grandparents
    grandparent_names = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in subfolders]

    # prepare dictionaries to store the geometry, tissue, and material fill status for each object
    classes = {}
    classes['hollow_cylinder1'] = {'geometry': 'cylinder', 
                                   'material_fill': 'hollow', 
                                   'tissue': ['water', 'muscle']}
    classes['hollow_cylinder2'] = {'geometry': 'cylinder', 
                                   'material_fill': 'hollow', 
                                   'tissue': ['water', 'liver', 'blood']}
    classes['hollow_sphere'] = {'geometry': 'sphere', 
                                'material_fill': 'hollow', 
                                'tissue': ['water', 'muscle', 'blood']}
    classes['solid_cone'] = {'geometry': 'cone', 
                             'material_fill': 'solid', 
                             'tissue': ['water', 'fatty_tissue']}
    classes['solid_cuboid'] = {'geometry': 'cuboid', 
                               'material_fill': 'solid', 
                               'tissue': ['water', 'bone']}
    classes['solid_cylinder'] = {'geometry': 'cylinder', 
                                 'material_fill': 'solid', 
                                 'tissue': ['water', 'bone']}
    classes['solid_sphere1'] = {'geometry': 'sphere', 
                                'material_fill': 'solid', 
                                'tissue': ['water', 'air']}
    classes['solid_sphere2'] = {'geometry': 'sphere', 
                                'material_fill': 'solid', 
                                'tissue': ['water', 'fatty_tissue']}
    classes['sphere_with_conical_spikes'] = {'geometry': 'sphere', 
                                             'material_fill': 'solid', 
                                             'tissue': ['water', 'fatty_tissue']}
    
    # create a dictionary called folder_names, with keys as the grandparent names and values as the subfolder paths
    folder_names = {}
    for grandparent_name, subfolder in zip(grandparent_names, subfolders):
        folder_names[grandparent_name] = subfolder

    # name the python script to be used later to create the images
    script_path = os.path.join(parent_folder, "run_simple_objects_US.py")
    # get the path of another script called "customPyFunction.py" in the same folder as this script
    custom_py_function_path = os.path.join(parent_folder, "customPyFunctions.py")
    # get the path of an xlsx file called "local_batch_uploads.xlsx" in the same folder as this script
    xlsx_file_path = os.path.join(parent_folder, "local_batch_uploads.xlsx")

    # get the current data and time
    current_time = datetime.now(pytz.timezone("America/Toronto")).strftime('DATE_%Y_%m_%d_TIME_%H_%M_%S')
    # create a folder called "dataset_current_time" in a folder called "dataset" in the grandparent folder
    dataset_folder = os.path.join(grandparent_folder, "dataset", f"dataset_{current_time}")
    os.makedirs(dataset_folder, exist_ok=True)
    # inside this folder, create three subfolders called "train", "val", and "test"
    train_set_folder = os.path.join(dataset_folder, "train")
    val_set_folder = os.path.join(dataset_folder, "val")
    test_set_folder = os.path.join(dataset_folder, "test")
    os.makedirs(train_set_folder, exist_ok=True)
    os.makedirs(val_set_folder, exist_ok=True)
    os.makedirs(test_set_folder, exist_ok=True)
    # inside each of these folders, create a subfolder called "class0"
    os.makedirs(os.path.join(train_set_folder, "class0"), exist_ok=True)
    os.makedirs(os.path.join(val_set_folder, "class0"), exist_ok=True)
    os.makedirs(os.path.join(test_set_folder, "class0"), exist_ok=True)

    for object_name in grandparent_names:
        create_images_for_objects(object_name,
                                  folder_names[object_name],
                                  dataset_folder,
                                  script_path,
                                  custom_py_function_path,
                                  xlsx_file_path)

    # Transfer all images from the object folders to the class0 folder in the train set
    class0_folder = os.path.join(train_set_folder, "class0")
    all_image_files = []
    for object_name in grandparent_names:
        image_files = glob.glob(os.path.join(dataset_folder, object_name, '*.png'))
        all_image_files.extend(image_files)

    # prepare an xlsx file called "original_class_info.xlsx" in class0_folder
    # which will store geometry, material fill, and tissue information for each image
    class_info_path = os.path.join(class0_folder, "original_class_info.xlsx")
    class_info = []
    # move the files to the class0 folder and prepare the class_info list
    for i, image_file in enumerate(all_image_files):
        # Get the object name from the image file path
        object_name = os.path.basename(os.path.dirname(image_file))
        # Get the class information for the object
        class_data = classes.get(object_name, {})
        # Append the image file name and class information to the class_info list
        class_info.append({
            "image": f"{i+1}.png" ,
            "geometry": class_data.get("geometry", ""),
            "material_fill": class_data.get("material_fill", ""),
            "tissue": class_data.get("tissue", [])
        })
        shutil.copy(image_file, os.path.join(class0_folder, f"{i+1}.png"))
    # Create a DataFrame from the class_info list
    df = pd.DataFrame(class_info)
    # Save the DataFrame to an Excel file
    df.to_excel(class_info_path, index=False)

    # Delete all object folders after copying the images
    for object_name in grandparent_names:
        shutil.rmtree(os.path.join(dataset_folder, object_name), ignore_errors=True)

    # Now split the images into train, val, and test sets
    # 15% each for val and test, the rest remain in train_set
    # Select 15% random numbers in increasing order from 0 to len(all_image_files) - 1
    # first for the val set, and then for the test set
    total_images = len(all_image_files)
    val_indices = sorted(random.sample(range(total_images), k=int(total_images * 0.15)))
    remaining_indices = set(range(total_images)) - set(val_indices)
    test_indices = sorted(random.sample(list(remaining_indices), k=int(total_images * 0.15)))

    # Move the selected images to the val and test folders
    for idx in val_indices:
        src_image_path = os.path.join(class0_folder, f"{idx}.png")
        dest_image_path = os.path.join(val_set_folder, "class0", f"{idx}.png")
        shutil.move(src_image_path, dest_image_path)

    for idx in test_indices:
        src_image_path = os.path.join(class0_folder, f"{idx}.png")
        dest_image_path = os.path.join(test_set_folder, "class0", f"{idx}.png")
        shutil.move(src_image_path, dest_image_path)

    # # now remove the respective indices from the class_info.xlsx file
    # # and prepare new class_info.xlsx files for val and test sets
    # df_val = df.iloc[val_indices].reset_index(drop=True)
    # df_test = df.iloc[test_indices].reset_index(drop=True)
    # df_train = df.drop(val_indices + test_indices).reset_index(drop=True)
    # df_val.to_excel(os.path.join(val_set_folder, "class0", "class_info.xlsx"), index=False)
    # df_test.to_excel(os.path.join(test_set_folder, "class0", "class_info.xlsx"), index=False)
    # df_train.to_excel(os.path.join(train_set_folder, "class0", "class_info.xlsx"), index=False)

    print("Dataset creation completed successfully.")
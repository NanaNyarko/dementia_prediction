# Ex: python preprocess.py -csv oasis_cross-sectional.csv -p data/raw_train -mt train -combine_ad
import argparse
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import csv
import shutil

csv_file = ""
data_input_path = ""
img_extensions = [".nifti.img", ".nifti.hdr"]
model_type = ""
combine_moderate_dementia = (
    False  # Because we don't have a lot of training data, combine 1.0 and 2.0
)
# CSR values for one moderate classification
combine_all_dementia = (
    False  # Combine CSR values for .5, 1.0, and 2.0 into one dementia classification
)

def build_unclassified_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "unclassified_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.move(individual_file_path, new_file_path)
    print(f"Saved image {filename} to unclassified_dementia folder in {model_type}.")

def build_moderate_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "moderate_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.move(individual_file_path, new_file_path)
    print(f"Saved image {filename} to moderate_dementia folder in {model_type}.")

def build_slight_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "slight_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.move(individual_file_path, new_file_path)
    print(f"Saved image {filename} to slight_dementia folder in {model_type}.")

def build_mild_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "mild_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.move(individual_file_path, new_file_path)
    print(f"Saved image {filename} to mild_dementia folder in {model_type}.")


def build_no_dementia_folder(individual_file_path, output_type):
    print(individual_file_path)
    print(output_type)
    output_file_path = os.path.join("data", model_type, "no_dementia")
    print(output_file_path)
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.move(individual_file_path, new_file_path) #I used move because I had space constraints. you can use copy to keep original folder structure
    print(f"Saved image {filename} to no_dementia folder in {model_type}.")


def check_is_img(filename):
    is_img = False
    for ext in img_extensions:
        if ext in filename:
            is_img = True
    return is_img

def move_files_up_and_suffix(source_dir):
    # Get list of directories in the source directory
    dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Move files from each 'RAW' directory to its parent directory
    for dir_name in dirs:
        raw_dir = os.path.join(source_dir, dir_name, 'RAW')
        
        # Check if the 'RAW' directory exists
        if os.path.exists(raw_dir):
            files = os.listdir(raw_dir)
            
            # Move each file from 'RAW' directory to its parent directory with suffix
            for file in files:
                source_path = os.path.join(raw_dir, file)
                dest_filename = f"{dir_name}_{file}"
                dest_path = os.path.join(source_dir, dir_name, dest_filename)
                shutil.move(source_path, dest_path)
                print(f"Moved '{file}' from '{raw_dir}' to '{os.path.join(source_dir, dir_name)}' with suffix '{dir_name}_'")
            
            # Remove the now empty 'RAW' directory
            os.rmdir(raw_dir)
            print(f"Removed empty directory '{raw_dir}'")

def check_against_csv(each, subfolder_name, subfolder_path):
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for record in reader:
            if record["ID"] == each:
                print(record['ID'], subfolder_name, record['CDR'])
                for filename in os.listdir(subfolder_path):
                    if check_is_img(filename):
                        # output_type = subfolder_name
                        output_type = "RAW" #hardcoded value to raw because i am currently working with only RAW 
                        individual_file_path = os.path.join(subfolder_path, filename)
                        record_cdr = float(record["CDR"]) if record["CDR"] else -10000
                        print(individual_file_path, record['CDR'])
                        if record_cdr == 0.0:
                            build_no_dementia_folder(individual_file_path, output_type)
                        elif record_cdr == 0.5:
                            build_mild_dementia_folder(individual_file_path, output_type)
                        elif combine_moderate_dementia and record_cdr in [1.0, 2.0]:
                            build_moderate_dementia_folder(individual_file_path, output_type)
                        elif combine_all_dementia and record_cdr in [0.5, 1.0, 2.0]:
                            build_moderate_dementia_folder(individual_file_path, output_type)
                        elif not combine_moderate_dementia and not combine_all_dementia:
                            if record_cdr == 1.0:
                                build_slight_dementia_folder(individual_file_path, output_type)
                            elif record_cdr == 2.0:
                                build_moderate_dementia_folder(individual_file_path, output_type)
                        elif record_cdr == -10000:
                            build_unclassified_dementia_folder(individual_file_path, output_type)

def crossreference_files(data_input_path):
    # Check if data_input_path exists and is a directory
    if os.path.isdir(data_input_path):
        # Retrieve a list of all items (files and directories) in data_input_path
        all_items = os.listdir(data_input_path)
        # Iterate over each item in data_input_path
        for item in all_items:
            # Construct the full path to the item
            item_path = os.path.join(data_input_path, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                # Print the name of the directory
                print(item)
                # Pass the directory name and path for further processing
                check_against_csv(item, item, item_path)
            else:
                # Skip files and print a message if encountered
                print(f"Skipping file: {item_path}")
    else:
        print(f"Error: {data_input_path} is not a valid directory.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-csv",
        "--csv",
        required=True,
        help="CSV file (script is currently formatted for cross-sectional records from Oasis)",
    )
    ap.add_argument(
        "-p",
        "--path",
        required=True,
        help="folder path containing input data for preprocessing",
    )
    ap.add_argument(
        "-mt",
        "--model_type",
        required=True,
        help="type of model to output preprocessed data (train, validation, or test)",
    )
    ap.add_argument(
        "-combine_md",
        "--combine_moderate_dementia",
        required=False,
        help="combines CDR of 1.0 and 2.0 into moderate classification",
        action="store_true",
    )
    ap.add_argument(
        "-combine_ad",
        "--combine_all_dementia",
        required=False,
        help="combines CDR of .5, 1.0, and 2.0 into moderate classification",
        action="store_true",
    )

    args = vars(ap.parse_args())
    csv_file = args["csv"]
    data_input_path = args["path"]
    model_type = args["model_type"]
    combine_moderate_dementia = args["combine_moderate_dementia"]
    combine_all_dementia = args["combine_all_dementia"]
    print("Lets prepare the image names before crossreference")
    move_files_up_and_suffix(data_input_path)
    print("Now preparing data from " + data_input_path + " " + "for training models.")
    crossreference_files(data_input_path)

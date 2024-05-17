import argparse
import csv
import os
import shutil
import nibabel as nib
import numpy as np
import cv2

csv_file = ""
data_input_path = ""
img_extensions = [".jpg"]
model_type = ""
combine_moderate_dementia = False
combine_all_dementia = False

def resize_nifti_image(nifti_path, output_size=(224, 224, 3)):
    print("NP ", nifti_path)
    # Load the NIfTI image
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Print the shape of the input data for debugging
    print("Input data shape:", data.shape)

    # Check if the input data is 4D (with the last dimension as 1)
    if data.ndim == 4 and data.shape[-1] == 1:
        # Reshape the data to remove the singleton dimension
        data = np.squeeze(data, axis=-1)

    # Check if the input data is 3D
    if data.ndim != 3:
        print(f"Error: Input data is not 3D. Skipping '{nifti_path}'.")
        return None

    # Resize the image data to the desired output size
    resized_data = np.zeros(output_size, dtype=data.dtype)
    for i in range(min(data.shape[-1], output_size[-1])):
        resized_data[..., i] = cv2.resize(data[..., i], (output_size[0], output_size[1]))

    # Create a new NIfTI image with the resized data
    resized_img = nib.Nifti1Image(resized_data, img.affine)
    
    # Save the resized image
    output_path = nifti_path[:-4] + '.nii'
    nib.save(resized_img, output_path)
    print(f"Resized image saved to '{output_path}'.")

    return resized_img

# Function to load NIfTI image and save slices as JPEG
def convert_nifti_to_jpg(input_path):
    print(f"Converting NIfTI image '{input_path}' to JPEG...")

    # Load NIfTI image
    nifti_img = nib.load(input_path)
    img_data = np.array(nifti_img.get_fdata())

    # Normalize pixel values to [0, 255]
    img_data = ((img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255).astype(np.uint8)

    # Create output directory if it doesn't exist
    print(f"input path ",input_path)
    
    filename = input_path.split('.', 1)[0] + '_slice.jpg'
    print("filename",filename)
    output_dir = os.path.dirname(input_path.split('.', 1)[0])
    print("output dir ", output_dir)
    print("desired ", input_path.split('.', 1)[0] + f"_slice.jpg")

    
    # Save each slice as JPEG
    for i in range(img_data.shape[2]):
        slice_img = img_data[:, :, i]
        cv2.imwrite(input_path.split('.', 1)[0] + f"_slice_{i}.jpg", slice_img)
    
    print(f"NIfTI image '{input_path}' converted to JPEG successfully.")

def convert_img_to_nii(nifti_img_path):
    print(f"Converting image '{nifti_img_path}' to .nii format...")
    
    # Check if the input file exists
    if not os.path.exists(nifti_img_path):
        print(f"Error: File '{nifti_img_path}' does not exist.")
        return None

    # Check if the input file has the '.nifti.img' extension
    if not nifti_img_path.endswith('.nifti.img'):
        print(f"Error: File '{nifti_img_path}' is not a '.nifti.img' file.")
        return None

    # Get the corresponding '.nii' file path
    nifti_nii_path = nifti_img_path.split('.', 1)[0] + '.nii'
    print("ok how dd ", nifti_nii_path)
    try:
        # Load the NIfTI image
        img = nib.load(nifti_img_path)
        
        # Save as NIfTI-1 format
        nib.save(img, nifti_nii_path)
        
        print(f"Converted '{nifti_img_path}' to '{nifti_nii_path}'.")
        return nifti_nii_path

    except Exception as e:
        print(f"Error: Failed to convert '{nifti_img_path}' to '.nii' format:", e)
        return None

def preprocess_nifti_images(input_dir, output_dir, target_size=(224, 224, 3)):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each directory and file in the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            # Check if the file is a '.nifti.img' file
            if filename.endswith('.nifti.img'):
                # Convert the '.nifti.img' file to '.nii' format
                print("about to convert to .img")
                nifti_nii_path = convert_img_to_nii(os.path.join(root, filename))
                if nifti_nii_path:
                    # Convert .nii to jpg
                    convert_nifti_to_jpg(nifti_nii_path)
                    # Resize the '.nii' file and save it to the output directory
                    resize_nifti_image(nifti_nii_path, target_size)
                    # output_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.nii')
                    # print(f"output path {output_path}")
                    # nib.save(resized_img, output_path)

    print("Preprocessing completed.")



def build_unclassified_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "unclassified_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.copy(individual_file_path, new_file_path)
    print(f"Saved image {filename} to unclassified_dementia folder in {model_type}.")

def build_moderate_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "moderate_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.copy(individual_file_path, new_file_path)
    print(f"Saved image {filename} to moderate_dementia folder in {model_type}.")

def build_slight_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "slight_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.copy(individual_file_path, new_file_path)
    print(f"Saved image {filename} to slight_dementia folder in {model_type}.")

def build_mild_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "mild_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.copy(individual_file_path, new_file_path)
    print(f"Saved image {filename} to mild_dementia folder in {model_type}.")

def build_no_dementia_folder(individual_file_path, output_type):
    output_file_path = os.path.join("data", model_type, "no_dementia")
    os.makedirs(output_file_path, exist_ok=True)
    filename = os.path.basename(individual_file_path)
    new_file_path = os.path.join(output_file_path, filename)
    shutil.copy(individual_file_path, new_file_path)
    print(f"Saved image {filename} to no_dementia folder in {model_type}.")

def check_is_img(filename):
    is_img = False
    for ext in img_extensions:
        if ext in filename:
            is_img = True
    return is_img

def move_files_up_and_suffix(source_dir):
    dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for dir_name in dirs:
        raw_dir = os.path.join(source_dir, dir_name, 'RAW')

        if os.path.exists(raw_dir):
            files = os.listdir(raw_dir)

            for file in files:
                source_path = os.path.join(raw_dir, file)
                dest_filename = f"{dir_name}_{file}"
                dest_path = os.path.join(source_dir, dir_name, dest_filename)
                shutil.move(source_path, dest_path)
                print(f"Moved '{file}' from '{raw_dir}' to '{os.path.join(source_dir, dir_name)}' with suffix '{dir_name}'")

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
                        output_type = "RAW"
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
    if os.path.isdir(data_input_path):
        all_items = os.listdir(data_input_path)
        for item in all_items:
            item_path = os.path.join(data_input_path, item)
            if os.path.isdir(item_path):
                print(item)
                check_against_csv(item, item, item_path)
            else:
                print(f"Skipping file: {item_path}")

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
    preprocess_nifti_images(data_input_path, os.path.join("data", model_type))
    crossreference_files(data_input_path)

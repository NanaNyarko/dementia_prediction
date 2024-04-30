import os
import nibabel as nib

# Input directory containing ".img" files
input_dir = "data/train/no_dementia_unconverted"

# Output directory for NIfTI files
output_dir = "data/train/no_dementia"

# Iterate over ".img" files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".nifti.img"):
        img_path = os.path.join(input_dir, filename)
        img = nib.load(img_path)
        
        # Construct output filename by replacing ".img" with ".nii"
        output_filename = filename.replace(".nifti.img", ".nii")
        output_path = os.path.join(output_dir, output_filename)
        
        # Save NIfTI file
        nib.save(img, output_path)
        print(output_path)

print("completed")
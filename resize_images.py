import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Input directory containing NIfTI files
input_dir = "sample_data/train/RAW/no_dementia"

# Output directory for processed images
output_dir = "sample_data/train/RAW/no_dementia_processed"

# Define the size to which the images will be resized
new_shape = (224, 224, 3)  # Change dimensions as needed

# Iterate over NIfTI files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".nii"):
        img_path = os.path.join(input_dir, filename)
        img = nib.load(img_path)
        img_data = img.get_fdata()

        # Resize image to new shape
        resize_factor = [new_dim / old_dim for new_dim, old_dim in zip(new_shape, img_data.shape)]
        resize_factor.extend([1] * (img_data.ndim - len(resize_factor)))  # Extend to match rank
        resized_img = zoom(img_data, resize_factor, mode='nearest')

        # Flatten image to one dimension
        flattened_img = resized_img.flatten()

        # Normalize voxel values to range [0, 1]
        normalized_img = flattened_img.astype(np.float32) / np.max(flattened_img)

        # Reshape the normalized image
        processed_img = normalized_img.reshape(new_shape)

        # Construct output filename
        output_filename = filename.replace(".nii", "_processed.nii")
        output_path = os.path.join(output_dir, output_filename)

        # Save processed image
        processed_nifti = nib.Nifti1Image(processed_img, img.affine)
        nib.save(processed_nifti, output_path)

        print("Processed:", output_path)

print("Completed")
import nibabel as nib
from scipy import ndimage
import numpy as np
import os

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
img = nib.load(data_path + "Spine.nii.gz")
data = img.get_fdata()

# Fill holes process

# Set up the kernel for the fill process and the parameters
n = 3
kernel = np.array([
                         [[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]],

                         [[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]],

                         [[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]]
                        ],
                  dtype=np.uint8)

# dilation, fill holes and erosion
modified_data = ndimage.binary_dilation(data, iterations=n)
modified_data = ndimage.binary_fill_holes(modified_data, structure=kernel).astype(int)
final_data = ndimage.binary_erosion(modified_data, iterations=n)


# Save the modified segmentation as a NIfTI file
modified_img = nib.Nifti1Image(final_data, img.affine, img.header)
nib.save(modified_img, os.path.join(data_path, 'show_results_n=0.nii.gz'))

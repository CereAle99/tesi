import nibabel as nib
from scipy import ndimage
import numpy as np
import os

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
original_file = nib.load(data_path + "Spine.nii.gz")


# Fill holes function
def fill_spinal_holes(input_nifti, n_dilations=3, dim=3):
    """

    Args:
        input_nifti:
        dim:
        n_dilations:

    Returns:

    """

    image = input_nifti.get_fdata()
    kernel = np.zeros((dim, dim, dim), dtype=np.uint8)
    kernel[:, dim // 2, :] = 1

    image = ndimage.binary_dilation(image, iterations=n_dilations)
    image = ndimage.binary_fill_holes(image, structure=kernel).astype(int)
    final_image = ndimage.binary_erosion(image, iterations=n_dilations)

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(final_image, input_nifti.affine, input_nifti.header)

    return final_nifti


# Fill the holes in the image
modified_file = fill_spinal_holes(original_file, 3)

# Save the modified segmentation as a NIfTI file
nib.save(modified_file, os.path.join(data_path, 'show_results_n=3.nii.gz'))

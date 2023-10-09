import nibabel as nib
from scipy import ndimage
import numpy as np
import os

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
img = nib.load(data_path + "Spine.nii.gz")
image = img.get_fdata()


# Fill holes function

def fill_spinal_holes(input_data, n_dilations=1, dim=3):
    """

    Args:
        input_data:
        dim:
        n_dilations:

    Returns:

    """
    data = input_data
    kernel = np.zeros((dim, dim, dim), dtype=np.uint8)
    kernel[:, dim // 2, :] = 1

    data = ndimage.binary_dilation(data, iterations=n_dilations)
    data = ndimage.binary_fill_holes(data, structure=kernel).astype(int)
    final_data = ndimage.binary_erosion(data, iterations=n_dilations)

    return final_data


# Fill the holes in the image
final_image = fill_spinal_holes(image, 3)

# Save the modified segmentation as a NIfTI file
modified_img = nib.Nifti1Image(final_image, img.affine, img.header)
nib.save(modified_img, os.path.join(data_path, 'show_results_n=0.nii.gz'))

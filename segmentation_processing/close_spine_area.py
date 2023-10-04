import nibabel as nib
from scipy import ndimage
import numpy as np
import os
from fill_spine import fill_spinal_holes

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
img = nib.load(data_path + "Spine.nii.gz")
image = img.get_fdata()


def dilate_spine(data, iterations=1):
    """

    Args:
        data:
        iterations:

    Returns:

    """

    data = fill_spinal_holes(data, 3)
    final_data = ndimage.binary_dilation(data, iterations=iterations)
    return final_data


final_image = dilate_spine(image, 3)

# Save the modified segmentation as a NIfTI file
modified_img = nib.Nifti1Image(final_image, img.affine, img.header)
nib.save(modified_img, os.path.join(data_path, 'close_spine_iter=1.nii.gz'))

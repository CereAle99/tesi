import nibabel as nib
from scipy import ndimage
import os
from fill_spine import fill_spinal_holes

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation
original_file = nib.load(data_path + "Spine.nii.gz")


def dilate_spine(input_nifti, iterations=3, fill=False):
    """

    Args:
        input_nifti:
        iterations:
        fill:

    Returns:

    """
    if fill:
        fill_nifti = fill_spinal_holes(input_nifti, 3)
    else:
        fill_nifti = input_nifti

    image = fill_nifti.get_fdata()
    final_image = ndimage.binary_dilation(image, iterations=iterations)

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(final_image, input_nifti.affine, input_nifti.header)

    return final_nifti


modified_file = dilate_spine(original_file, 3, True)

# Save the modified segmentation as a NIfTI file
nib.save(modified_file, os.path.join(data_path, 'close_spine_iter=3.nii.gz'))

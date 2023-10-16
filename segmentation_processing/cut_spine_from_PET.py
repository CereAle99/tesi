import nibabel as nib
import numpy as np
import os
from resizePET_Ale import pet_ct_real_dim_compatible, pet_compatible_to_ct

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'
image_path = data_path + 'test_PET/MPC_2_20110413/'

# Load the segmentation NIfTI file
spine = nib.load(data_path + "Spine.nii.gz")

# Load the segmentation NIfTI file
image_obj = nib.load(image_path + "PT.nii")


def cut_spine_shape(input_image, mask):  # whether the dim are the same will be treated when resize function is complete
    """

    Args:
        input_image:
        mask:

    Returns:

    """

    # Make PET image and spine segmentation image compatibles
    resized_pet, resized_mask = pet_compatible_to_ct(input_image, mask, True)

    # Put the segmentation into a numpy array
    spine_mask = resized_mask.get_fdata()

    # Put the image into a numpy array
    image = resized_pet.get_fdata()

    # Make the mask binary and multiply it to the input image
    mask_01 = spine_mask
    mask_01[mask_01 != 41] = 0
    mask_01[mask_01 == 41] = 1
    cut_image = image * mask_01

    # Save cut image in a NIfTI file
    cut_file = nib.Nifti1Image(cut_image, resized_pet.affine, resized_pet.header)

    return cut_file


# Cut the image
final_image = cut_spine_shape(image_obj, spine)

# Save the modified segmentation as a NIfTI file
nib.save(final_image, os.path.join(data_path, 'spine_PET.nii'))

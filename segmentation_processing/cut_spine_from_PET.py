import nibabel as nib
import numpy as np
import os
from resizePET_Ale import pet_ct_real_dim_compatible, pet_compatible_to_ct

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'
image_path = data_path + 'test_PET/MPC_2_20110413/'

# Load the segmentation NIfTI file
ct = nib.load(image_path + "CT.nii")

# Load the segmentation NIfTI file
image_obj = nib.load(image_path + "PT.nii")


def cut_spine_shape(input_image, mask):  # whether the dim are the same will be treated when resize function is complete
    """

    Args:
        input_image:
        mask:

    Returns:

    """

    # Put the segmentation into a numpy array
    spine_mask = mask.get_fdata()

    # Put the image into a numpy array
    image = input_image.get_fdata()

    # Make the mask composed by zero and ones and multiply it to the input image
    mask_01 = spine_mask
    mask_01[mask_01 < 41.] = 0
    mask_01[mask_01 >= 41.] = 1
    cut_image = image * mask_01

    return cut_image


# Cut the image
# final_image = cut_spine_shape(image_obj, spine)


# Save the modified segmentation as a NIfTI file
# modified_img = nib.Nifti1Image(final_image, image_obj.affine, image_obj.header)
# nib.save(modified_img, os.path.join(data_path, 'spine_PET.nii'))

# pet_image, ct_image = pet_ct_real_dim_compatible(image_obj, spine, True)
pet_image, ct_image = pet_compatible_to_ct(image_obj, ct, False)

nib.save(pet_image, os.path.join(data_path, 'PET_resized_to_ct.nii'))
nib.save(ct_image, os.path.join(data_path, 'ct_resized_to_pet.nii'))

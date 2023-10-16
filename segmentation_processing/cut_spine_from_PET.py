import nibabel as nib
import numpy as np
import os
from resizePET_Ale import pet_compatible_to_ct

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'
image_path = data_path + 'test_PET/MPC_2_20110413/'

# Load the segmentation NIfTI file
spine = nib.load(data_path + "Spine.nii.gz")

# Load the segmentation NIfTI file
image_obj = nib.load(image_path + "PT.nii")


def cut_spine_shape(input_nifti, mask, segmentation_value=1):
    """

    Args:
        input_nifti:
        mask:
        segmentation_value:

    Returns:

    """

    # Make PET image and spine segmentation image compatibles
    resized_pet, resized_mask = pet_compatible_to_ct(input_nifti, mask, segmentation_value)

    # Put the segmentation into a numpy array
    spine_mask = resized_mask.get_fdata()

    # Put the image into a numpy array
    image = resized_pet.get_fdata()

    # Cut the PET image
    cut_image = image * spine_mask

    # Save cut image in a NIfTI file
    cut_file = nib.Nifti1Image(cut_image, resized_pet.affine, resized_pet.header)

    return cut_file


# Cut the image
final_image = cut_spine_shape(image_obj, spine)

# Save the modified segmentation as a NIfTI file
nib.save(final_image, os.path.join(data_path, 'spine_PET.nii'))

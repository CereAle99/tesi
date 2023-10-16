import nibabel as nib
import numpy as np
import os
from close_spine_area import dilate_spine

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
original_file = nib.load(data_path + "Spine.nii.gz")


def spine_as_cylinder(input_nifti, close_spine=3):
    """

    Args:
        input_nifti:
        close_spine:

    Returns:

    """

    if close_spine != 0:
        dilated_nifti = dilate_spine(input_nifti, close_spine)
    else:
        dilated_nifti = input_nifti

    image = dilated_nifti.get_fdata()

    [i_max, i_min, j_max, j_min, k_max, k_min] = [0, 999, 0, 999, 0, 999]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if image[i, j, k] != 0:
                    i_max, i_min = np.maximum(i_max, i), np.minimum(i_min, i)
                    j_max, j_min = np.maximum(j_max, j), np.minimum(j_min, j)
                    k_max, k_min = np.maximum(k_max, k), np.minimum(k_min, k)

    cylinder_center = np.array([(i_max + i_min)*0.5, (j_max + j_min)*0.5])
    cylinder_radius = np.maximum((i_max - i_min)*0.5, (j_max - j_min)*0.5)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            point = np.array([i, j])
            if np.linalg.norm(cylinder_center - point) < cylinder_radius:
                image[i, j, k_min:k_max+1] = 1

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(image, input_nifti.affine, input_nifti.header)

    return final_nifti


modified_file = spine_as_cylinder(original_file, 3)

# Save the modified segmentation as a NIfTI file
nib.save(modified_file, os.path.join(data_path, 'spine_cylinder.nii.gz'))

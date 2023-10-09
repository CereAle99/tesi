import nibabel as nib
import numpy as np
import os
from close_spine_area import dilate_spine

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
img = nib.load(data_path + "Spine.nii.gz")
image = img.get_fdata()


def spine_as_cylinder(input_data, close_spine=0):
    """

    Args:
        input_data:
        close_spine:

    Returns:

    """

    data = input_data

    if close_spine != 0:
        data = dilate_spine(data, close_spine)

    [i_max, i_min, j_max, j_min, k_max, k_min] = [0, 999, 0, 999, 0, 999]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i, j, k] != 0:
                    i_max, i_min = np.maximum(i_max, i), np.minimum(i_min, i)
                    j_max, j_min = np.maximum(j_max, j), np.minimum(j_min, j)
                    k_max, k_min = np.maximum(k_max, k), np.minimum(k_min, k)

    cylinder_center = np.array([(i_max + i_min)*0.5, (j_max + j_min)*0.5])
    cylinder_radius = np.maximum((i_max - i_min)*0.5, (j_max - j_min)*0.5)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            point = np.array([i, j])
            if np.linalg.norm(cylinder_center - point) < cylinder_radius:
                data[i, j, k_min:k_max+1] = 41.0

    final_data = data
    return final_data


final_image = spine_as_cylinder(image, 3)

# Save the modified segmentation as a NIfTI file
modified_img = nib.Nifti1Image(final_image, img.affine, img.header)
nib.save(modified_img, os.path.join(data_path, 'spine_cylinder.nii.gz'))

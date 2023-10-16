import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os


def pet_ct_real_dim_compatible(pet_nifti, ct_nifti, segmentation=False):
    """

    Args:
        pet_nifti:
        ct_nifti:
        segmentation:

    Returns:

    """

    # Load the nifti files header and image array
    pet_header = pet_nifti.header
    ct_header = ct_nifti.header
    pet_affine = pet_nifti.affine
    ct_affine = ct_nifti.affine
    pet_image = pet_nifti.get_fdata()
    ct_image = ct_nifti.get_fdata()

    # PET resizing ratio
    pixdim1 = np.array(pet_header['pixdim'][1:4])
    pixdim2 = np.array(ct_header['pixdim'][1:4])
    pixdim1[2] = 1
    pixdim2[2] = 1

    # PET resizing
    pet_image = zoom(pet_image, zoom=pixdim1)
    ct_image = zoom(ct_image, zoom=pixdim2, order=1)

    # Managing the offset
    pet_header['qoffset_x'] = (pet_header['qoffset_x']
                               + (pet_header['pixdim'][1] / 2)
                               - (pet_header['pixdim'][1] / pixdim1[0]) / 2)
    pet_header['qoffset_y'] = (pet_header['qoffset_y']
                               + (pet_header['pixdim'][2] / 2)
                               - (pet_header['pixdim'][2] / pixdim1[1]) / 2)

    # PET header fixing
    pet_header['dim'][1:4] = pet_image.shape
    pet_affine[0, 0] = -pet_header['pixdim'][1] / pixdim1[0]
    pet_affine[1, 1] = -pet_header['pixdim'][2] / pixdim1[1]
    pet_affine[2, 2] = pet_header['pixdim'][3] / pixdim1[2]
    pet_affine[0, 3] = pet_header['qoffset_x']
    pet_affine[1, 3] = pet_header['qoffset_y']
    pet_affine[2, 3] = pet_header['qoffset_z']

    # CT header fixing
    ct_header['dim'][1:4] = pet_image.shape
    ct_affine[0, 0] = -ct_header['pixdim'][1] / pixdim2[0]
    ct_affine[1, 1] = -ct_header['pixdim'][2] / pixdim2[1]
    ct_affine[2, 2] = ct_header['pixdim'][3] / pixdim2[2]
    ct_affine[0, 3] = pet_header['qoffset_x']
    ct_affine[1, 3] = pet_header['qoffset_y']
    ct_affine[2, 3] = pet_header['qoffset_z']

    # CT image resizing
    ct_image_resized = np.zeros(shape=pet_image.shape)
    side_x = (pet_image.shape[0]-ct_image.shape[0])//2
    side_y = (pet_image.shape[1]-ct_image.shape[1])//2
    side_z = (pet_image.shape[2]-ct_image.shape[2])//2
    center_x = ct_image.shape[0]
    center_y = ct_image.shape[1]
    center_z = ct_image.shape[2]
    ct_image_resized[side_x-1:side_x+center_x-1, side_y-1:side_y+center_y-1, side_z:side_z+center_z] = ct_image

    if segmentation:
        mask_01 = ct_image_resized
        mask_01[mask_01 > 0] = 41
        mask_01[mask_01 < 0] = 41
        ct_image_resized = mask_01

    # CT and PET NIfTI files assembled
    resized_pet = nib.Nifti1Image(pet_image, pet_affine, pet_header)
    resized_ct = nib.Nifti1Image(ct_image_resized, ct_affine, ct_header)
    return resized_pet, resized_ct


def pet_compatible_to_ct(pet_nifti, ct_nifti, segmentation=False):
    """

    Args:
        pet_nifti:
        ct_nifti:
        segmentation:

    Returns:

    """

    # Load the nifti files header and image array
    pet_header = pet_nifti.header
    ct_header = ct_nifti.header
    pet_affine = pet_nifti.affine
    ct_affine = ct_nifti.affine
    pet_image = pet_nifti.get_fdata()
    ct_image = ct_nifti.get_fdata()
    print(pet_header)
    print(ct_header)

    # PET resize ratio
    resize_ratio = pet_header['pixdim'][1:4] / ct_header['pixdim'][1:4]

    # PET resizing and his displacement
    pet_image = zoom(pet_image, zoom=resize_ratio)
    rest = np.array(pet_image.shape) - np.array(pet_header['dim'][1:4]) * np.array(resize_ratio)
    pixel_displacement = rest / np.array(pet_image.shape)
    print(pixel_displacement)

    # Managing the offset
    pet_header['qoffset_x'] = (pet_header['qoffset_x']
                               + (pet_header['pixdim'][1] / 2)
                               - (pet_header['pixdim'][1] / resize_ratio[0]) / 2)
    pet_header['qoffset_y'] = (pet_header['qoffset_y']
                               + (pet_header['pixdim'][2] / 2)
                               - (pet_header['pixdim'][2] / resize_ratio[1]) / 2)

    # PET header fixing
    pet_header['dim'][1:4] = pet_image.shape
    pet_affine[0, 0] = -(pet_header['pixdim'][1] / resize_ratio[0] - pixel_displacement[0])
    pet_affine[1, 1] = -(pet_header['pixdim'][2] / resize_ratio[1] - pixel_displacement[1])
    pet_affine[2, 2] = (pet_header['pixdim'][3] / resize_ratio[2] - pixel_displacement[2])
    pet_affine[0, 3] = pet_header['qoffset_x']
    pet_affine[1, 3] = pet_header['qoffset_y']
    pet_affine[2, 3] = pet_header['qoffset_z']

    # CT header fixing
    ct_header['dim'][1:4] = pet_image.shape
    ct_affine[0, 0] = -(ct_header['pixdim'][1] - pixel_displacement[0])
    ct_affine[1, 1] = -(ct_header['pixdim'][2] - pixel_displacement[1])
    ct_affine[2, 2] = (ct_header['pixdim'][3] - pixel_displacement[2])
    ct_affine[0, 3] = pet_header['qoffset_x']
    ct_affine[1, 3] = pet_header['qoffset_y']
    ct_affine[2, 3] = pet_header['qoffset_z']

    # CT image resizing
    ct_image_resized = np.zeros(shape=pet_image.shape)
    side_x = (pet_image.shape[0]-ct_image.shape[0])//2
    side_y = (pet_image.shape[1]-ct_image.shape[1])//2
    side_z = (pet_image.shape[2]-ct_image.shape[2])//2
    center_x = ct_image.shape[0]
    center_y = ct_image.shape[1]
    center_z = ct_image.shape[2]
    ct_image_resized[side_x:side_x+center_x, side_y:side_y+center_y, side_z:side_z+center_z] = ct_image

    if segmentation:
        mask_01 = ct_image_resized
        mask_01[mask_01 > 0] = 41
        mask_01[mask_01 < 0] = 41
        ct_image_resized = mask_01

    # CT and PET NIfTI files assembled
    resized_pet = nib.Nifti1Image(pet_image, pet_affine, pet_header)
    resized_ct = nib.Nifti1Image(ct_image_resized, ct_affine, ct_header)
    print(resized_pet.header)
    print(resized_ct.header)

    return resized_pet, resized_ct


if __name__ == "__main__":
    # All patients path
    current_directory = os.getcwd()
    data_path = current_directory + '/data/test_PET'

    for folder in os.listdir(data_path):
        print("Patient: ", folder)
        # Get label path
        patient_path = os.path.join(data_path, folder)
        image_path = os.path.join(patient_path, "PT.nii")

        # Load label
        img_obj = nib.load(image_path)
        img = img_obj.get_fdata()
        print("Initial shape: ", img.shape)
        print("Old header: \n", img_obj.header)

        # Evaluate the ratio of the images dimension
        CT_PET_ratio = np.array([512, 512, 237]) / np.array(img.shape)

        # Transform label
        img = zoom(img, zoom=CT_PET_ratio)
        print("Final shape: ", img.shape)

        # Modify file header and affine
        new_header = img_obj.header
        new_header['dim'] = [3, 512, 512, 237, 1, 1, 1, 1]
        new_header['pixdim'] = [
                                1.,
                                5.46875/CT_PET_ratio[0],
                                5.46875/CT_PET_ratio[1],
                                3.27/CT_PET_ratio[2],
                                0.,
                                0.,
                                0.,
                                0.
                               ]
        new_header['srow_x'] = [-5.46875/CT_PET_ratio[0], 0., 0., 347.26562]
        new_header['srow_y'] = [0., -5.46875/CT_PET_ratio[1], 0., 347.26562]
        new_header['srow_z'] = [0., 0., 3.27/CT_PET_ratio[2], -771.7]
        new_affine = np.array(
                              [
                                [-5.46875/CT_PET_ratio[0], 0., 0., 347.26562],
                                [0., -5.46875/CT_PET_ratio[1], 0., 347.26562],
                                [0., 0., 3.27/CT_PET_ratio[2], -771.7],
                                [0., 0., 0., 1.]
                              ]
                             )

        img_resized = nib.Nifti1Image(img, new_affine, new_header)
        print("New header: \n", img_resized.header)

        # Save label
        save_path = os.path.join(patient_path, "PET-Resized")
        os.makedirs(save_path, exist_ok=True)
        if os.path.isfile(save_path + '/PT.nii'):
            print("Saving PET")
            nib.save(img_resized, save_path + '/PT.nii')

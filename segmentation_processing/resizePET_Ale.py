import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os

if __name__ == "__main__":
    # All patients path
    path = "data/test_PET"

    for folder in os.listdir(path):
        print("Patient: ", folder)
        # Get label path
        patient_path = os.path.join(path, folder)
        image_path = os.path.join(patient_path, "PT.nii")

        # Load label
        img_obj = nib.load(image_path)
        img = img_obj.get_fdata()
        print("Initial shape: ", img.shape)
        print(img_obj.header)
        print(img_obj.affine)

        # Evaluate the ratio of the images dimension
        CT_PET_ratio = np.array([512, 512, 239]) / np.array(img.shape)

        # Transform label
        img = zoom(img, zoom=CT_PET_ratio)
        print("Final shape: ", img.shape)

        # Modify file header and affine
        new_header = img_obj.header
        new_header['dim'] = [3, 512, 512, 239, 1, 1, 1, 1]
        new_header['pixdim'] = [1.,
                                5.46875/CT_PET_ratio[0],
                                5.46875/CT_PET_ratio[1],
                                3.27/CT_PET_ratio[2],
                                0.,
                                0.,
                                0.,
                                0.]
        new_header['srow_x'] = [-5.46875/CT_PET_ratio[0], 0., 0., 347.26562]
        new_header['srow_y'] = [0., -5.46875/CT_PET_ratio[1], 0., 347.26562]
        new_header['srow_z'] = [0., 0., 3.27/CT_PET_ratio[2], -771.7]
        new_affine = np.array([
                                [-5.46875 / CT_PET_ratio[0], 0., 0., 347.26562],
                                [0., -5.46875 / CT_PET_ratio[1], 0., 347.26562],
                                [0., 0., 3.27 / CT_PET_ratio[2], -771.7],
                                [0., 0., 0., 1.]
                            ])

        img_resized = nib.Nifti1Image(img, new_affine, header=new_header)
        print(img_resized.header)

        # Save label
        save_path = os.path.join(patient_path, "PET-Resized")
        os.makedirs(save_path, exist_ok=True)
        if not os.path.isfile(save_path + '/PT.nii'):
            print("Saving PET")
            nib.save(img_resized, save_path + '/PT.nii')


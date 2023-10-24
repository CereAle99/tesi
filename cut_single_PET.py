import os
import nibabel as nib
from segmentation_processing.crop_spine_from_PET import crop_spine_shape

if __name__ == "__main__":

    # All patients path
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "data", "test_PET", "MPC_101_20130628")

    label_path = os.path.join(data_path, "Spine.nii.gz")
    pet_path = os.path.join(data_path, "PT.nii")

    segmentation_file = nib.load(label_path)
    pet_file = nib.load(pet_path)

    cut_pet = crop_spine_shape(pet_file, segmentation_file, "original", 41)

    nib.save(cut_pet, data_path + "/PT_pixdim_shift_fix.nii")

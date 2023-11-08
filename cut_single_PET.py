import os
import nibabel as nib
from segmentation_processing.crop_spine_from_PET import crop_spine_shape

if __name__ == "__main__":

    # All patients path
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    moose_path1 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_1"
    moose_path2 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_2"
    segmentations_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Segmentations"
    original_data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/sick_patients"
    current_path = os.getcwd()

    # next_patient_id =

    # Different patient
    patient_id = "MPC_3350_20210722"

    # data_path = os.path.join(current_path, "data", "test_PET", "3338")
    data_path = os.path.join(segmentations_path, patient_id)

    label_path = os.path.join(data_path, "Bones.nii.gz")
    pet_path = os.path.join(original_data_path, patient_id, "PT.nii")

    segmentation_file = nib.load(label_path)
    pet_file = nib.load(pet_path)

    cut_pet = crop_spine_shape(pet_file, segmentation_file, "cylinder", 15)

    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_cylinder.nii"))

    cut_pet = crop_spine_shape(pet_file, segmentation_file, "original", 15)

    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_original.nii"))

    cut_pet = crop_spine_shape(pet_file, segmentation_file, "dilation", 15)

    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_dilation.nii"))

    cut_pet = crop_spine_shape(pet_file, segmentation_file, "fill_holes", 15)

    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_fill_holes.nii"))

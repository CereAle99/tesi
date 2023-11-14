import os
import nibabel as nib
from segmentation_processing.crop_spine_from_PET import crop_spine_shape

if __name__ == "__main__":

    # All patients path
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    mm_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma"

    moose_path1 = mm_path + "/Original/Moose_output/moose_1"
    moose_path2 = mm_path + "/Original/Moose_output/moose_2"
    segmentations_path = mm_path + "/Original/Segmentations"
    original_data_path = mm_path + "/Original/PET-CT"
    cropping_sick_path = mm_path + "/spine_PET/sick_patients"
    cropping_healthy_path = mm_path + "/spine_PET/healthy_patients"
    pet_healthy = mm_path + "/Healthy/HEALTHY-PET-CT/FDG-PET-CT-Lesions"
    healthy_segmentations = mm_path + "/Healthy/moose_output/"
    current_path = os.getcwd()

    # patient_id = "PETCT_3cd49210eb"
    patient_id = "MPC_122_20141223"

    # Used paths
    # label_path = os.path.join(healthy_segmentations, patient_id, "moosez-clin_ct_bones_v1-2023-10-30-22-48-23",
    #                           "segmentations", "CT_Bones_V1_CT_4_gk_pv3_0000.nii.gz")
    # image_path = os.path.join(pet_healthy, patient_id, "8.000000-PET_corr.-68079_PET_corr._20021229083413_8.nii")
    # save_path = os.path.join(current_path, "data", "test_PET")

    # label_path = os.path.join(current_path, "data", "test_PET", patient_id, "Spine.nii.gz")
    # image_path = os.path.join(current_path, "data", "test_PET", patient_id, "PT.nii")
    # save_path = os.path.join(current_path, "data", "test_PET", patient_id)

    label_path = os.path.join(moose_path1, patient_id, "Bones.nii.gz")
    image_path = os.path.join(original_data_path, patient_id, "PT.nii")
    save_path = os.path.join(cropping_sick_path, patient_id)

    segmentation_file = nib.load(label_path)
    image_file = nib.load(image_path)

    # cut_pet = crop_spine_shape(image_file, segmentation_file, "cylinder", 15)
    #
    # nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_cylinder.nii"))

    cut_pet = crop_spine_shape(image_file, segmentation_file, "original", 41)

    nib.save(cut_pet, os.path.join(save_path, "PT_original_corrected_axis.nii"))

    # cut_pet = crop_spine_shape(image_file, segmentation_file, "dilation", 15)
    #
    # nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_dilation.nii"))
    #
    # cut_pet = crop_spine_shape(image_file, segmentation_file, "fill_holes", 15)
    #
    # nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_fill_holes.nii"))

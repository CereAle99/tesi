import os
import nibabel as nib
from segmentation_processing.crop_spine_from_CT import crop_spine_from_ct
from segmentation_processing.crop_spine_from_PET import crop_spine_shape

if __name__ == "__main__":

    # All patients path
    shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
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


    patient_id = "MPC_296_20190708"

    # Used paths

    

    # label_path = os.path.join(healthy_segmentations, patient_id, "moosez-clin_ct_bones_v1-2023-10-31-17-30-38",
    #                           "segmentations", "CT_Bones_V1_CT_4_gk_pv3_0000.nii.gz")
    # image_path = os.path.join(pet_healthy, patient_id, "5.000000-PET_corr.-91812_PET_corr._20021019123810_5.nii")
    # save_path = cropping_healthy_path

    # label_path = os.path.join(moose_path1, patient_id, "MOOSE-P1/labels/sim_space/similarity-space", "Spine.nii.gz")
    # image_path = os.path.join(original_data_path, patient_id, "PT2.nii")
    # save_path = cropping_sick_path

    # label_path = os.path.join(moose_path2, patient_id, "Bones.nii.gz")
    # image_path = os.path.join(original_data_path, patient_id, "PT.nii")
    # save_path = cropping_sick_path

    label_path = os.path.join(moose_path2, patient_id, "moosez-clin_ct_bones_v1-2023-10-03-17-54-37", "segmentations", "CT_Bones_V1_CT_0000.nii.gz")
    image_path = os.path.join(original_data_path, patient_id, "PT2.nii")
    save_path = cropping_sick_path

    segmentation_file = nib.load(label_path)
    image_file = nib.load(image_path)
    cut_pet = crop_spine_shape(image_file, segmentation_file, "cylinder", 15)
    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_cylinder.nii.gz"))

    segmentation_file = nib.load(label_path)
    image_file = nib.load(image_path)
    cut_pet = crop_spine_shape(image_file, segmentation_file, "original", 15)
    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_original.nii.gz"))

    segmentation_file = nib.load(label_path)
    image_file = nib.load(image_path)
    cut_pet = crop_spine_shape(image_file, segmentation_file, "dilation", 15)
    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_dilation.nii.gz"))

    segmentation_file = nib.load(label_path)
    image_file = nib.load(image_path)
    cut_pet = crop_spine_shape(image_file, segmentation_file, "fill_holes", 15)
    nib.save(cut_pet, os.path.join(save_path, patient_id, "PT_fill_holes.nii.gz"))

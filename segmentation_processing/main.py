import os
import nibabel as nib
from crop_spine_from_PET import crop_spine_shape

if __name__ == "__main__":
    # All patients path
    moose_path1 = ("/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,"
                   "volume=Genomed/Genomed4All_Data/MultipleMieloma/Segmentations/moose_1")
    moose_path2 = ("/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,"
                   "volume=Genomed/Genomed4All_Data/MultipleMieloma/Segmentations/moose_2")
    patient_path = ("/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,"
                    "volume=Genomed/Genomed4All_Data/MultipleMieloma/PET-CT")
    label_folder = "labels/sim_space/similarity-space"

    # Get saving directory path
    save_path = ("/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,"
                 "volume=Genomed/Genomed4All_Data/MultipleMieloma/Segmentations")

    # For each patient in folder moose_1
    for folder in os.listdir(moose_path1):
        print("Patient: ", folder)

        # Get label and PET path
        patient_path = os.path.join(moose_path1, folder)
        label_path = os.path.join(patient_path, "MOOSE-" + folder, label_folder, "Spine.nii.gz")
        pet_path = os.path.join(patient_path, folder, "PT.nii")

        # Load label
        segmentation_file = nib.load(label_path)
        pet_file = nib.load(pet_path)

        # Shape the PET image
        shapes = ["original", "fill_holes", "close_spine", "cylinder"]
        for function in shapes:
            if function == "original":
                label = 41
            else:
                label = 1
            cut_pet = crop_spine_shape(pet_file, segmentation_file, function, label)
            nib.save(cut_pet, os.path.join(save_path, folder, 'PT_' + function + ".nii"))

    # Have to put a way to find the path of each file because moose 2.0 changes the folder name everytime
    #
    #
    # for folder in os.listdir(moose_path2):
    #     print("Patient: ", folder)
    #
    #     # Get label and PET path
    #     patient_path = os.path.join(moose_path2, folder)
    #     label_path = os.path.join(patient_path, "MOOSE-" + folder, label_folder, "Spine.nii.gz")
    #     pet_path = os.path.join(patient_path, folder, "PT.nii")
    #
    #     # Load label
    #     segmentation_file = nib.load(label_path)
    #     pet_file = nib.load(pet_path)
    #
    #     # Shape the PET image
    #     shapes = ["original", "fill_holes", "close_spine", "cylinder"]
    #     for function in shapes:
    #         if function == "original":
    #             label = 41
    #         else:
    #             label = 1
    #         cut_pet = crop_spine_shape(pet_file, segmentation_file, function, label)
    #         nib.save(cut_pet, os.path.join(save_path, folder, 'PT_' + function + ".nii"))

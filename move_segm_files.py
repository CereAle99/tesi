import os
import nibabel as nib
import numpy as np
import pandas as pd
import copy
from segmentation_processing.binarize import binarize
from segmentation_processing.spine_outofthe_cylinder import spine_as_cylinder
from segmentation_processing.fill_spine import fill_spinal_holes
from segmentation_processing.close_spine_area import dilate_spine


if __name__ == "__main__":

    # All patients path
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
    moose_path1 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_1"
    moose_path2 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_2"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/sick_patients"
    # save_path = os.path.join(current_path, "data", "test_PET")

    # For each patient in folder moose_1
    for patient_id in os.listdir(moose_path1):
        print("Patient: ", patient_id)
    
        if patient_id != "MPC_249_20130306":
            continue


        try:
            # Get label path
            label_folder = "labels/sim_space/similarity-space"
            patient_path = os.path.join(moose_path1, patient_id)
            # label_path = os.path.join(patient_path, "MOOSE-" + patient_id, label_folder, "Spine.nii.gz")
            label_path = os.path.join(patient_path, "Bones_CT.nii.gz")

            # Load label


            # Load label and perform the shaping
            mask = nib.load(label_path)
            mask = binarize(mask, 41)
            segmentation = copy.deepcopy(mask)
            spine_filled = fill_spinal_holes(segmentation, 3, 3)
            segmentation = copy.deepcopy(mask)
            spine_dilated = dilate_spine(segmentation, 3, True)
            segmentation = copy.deepcopy(mask)
            spine_cylinder = spine_as_cylinder(segmentation, 3)
            segmentation = copy.deepcopy(mask)

            # Save cropped PET
            save_dir = os.path.join(save_path, patient_id)
            nib.save(segmentation, save_dir + "/mask_CT_original.nii.gz")
            nib.save(spine_filled, save_dir + "/mask_CT_fill_holes.nii.gz")
            nib.save(spine_dilated, save_dir + "/mask_CT_dilation.nii.gz")
            nib.save(spine_cylinder, save_dir + "/mask_CT_cylinder.nii.gz")
            print("Saved: ", patient_id)
      
        except FileNotFoundError:
            print("FileNotFoundError for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/move_segm_log.txt", "a") as file:
                file.write(f"{patient_id}: FileNotFoundError\n")
            continue
    
        except StopIteration:
            print("StopIteration for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/move_segm_log.txt", "a") as file:
                file.write(f"{patient_id}: StopIteration\n")
            continue
    
        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/move_segm_log.txt", "a") as file:
                file.write(f"{patient_id}: Unknown error ({e})\n")
            continue
    
    # For each patient in folder moose_2
    for patient_id in os.listdir(moose_path2):
        
        if patient_id != "MPC_249_20130306":
            continue

        print("Patient: ", patient_id)
    
        try:
    
            # # Find the label folder for moose 2.0
            start_seq = "moosez"
            label_folder = [d for d in os.listdir(patient_path) if d.startswith(start_seq)]
            label_path = os.path.join(patient_path, label_folder[0], "segmentations", "CT_Bones_V1_CT_0000.nii.gz")
            # label_path = os.path.join(patient_path, "Bones.nii.gz")
    
            # Load label
            mask = nib.load(label_path)

            # Perform the shaping
            mask = nib.load(label_path)
            mask = binarize(mask, 15)
            spine_filled = fill_spinal_holes(mask, 3, 3)
            mask = nib.load(label_path)
            mask = binarize(mask, 15)
            spine_dilated = dilate_spine(mask, 3, True)
            mask = nib.load(label_path)
            mask = binarize(mask, 15)
            spine_cylinder = spine_as_cylinder(mask, 3)
            mask = nib.load(label_path)
            mask = binarize(mask, 15)

            # Save cropped PET
            save_dir = os.path.join(save_path, patient_id)
            nib.save(mask, save_dir + "/mask_CT_original.nii.gz")
            nib.save(spine_filled, save_dir + "/mask_CT_fill_holes.nii.gz")
            nib.save(spine_dilated, save_dir + "/mask_CT_dilation.nii.gz")
            nib.save(spine_cylinder, save_dir + "/mask_CT_cylinder.nii.gz")
            print("Saved: ", patient_id)
    
        except FileNotFoundError:
            print("FileNotFoundError for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/move_segm_log.txt", "a") as file:
                file.write(f"{patient_id}: FileNotFoundError\n")
    
        except StopIteration:
            print("StopIteration for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/move_segm_log.txt", "a") as file:
                file.write(f"{patient_id}: StopIteration\n")
    
        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/move_segm_log.txt", "a") as file:
                file.write(f"{patient_id}: Unknown error ({e})\n")
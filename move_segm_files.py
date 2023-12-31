import os
import nibabel as nib
from lib.segmentation_processing.segm_binarize import binarize
from lib.segmentation_processing import spine_as_cylinder
from lib.segmentation_processing import fill_spinal_holes
from lib.segmentation_processing import dilate_spine


if __name__ == "__main__":

    # All patients path
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    moose_path1 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_1"
    moose_path2 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_2"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/sick_patients"
    # save_path = os.path.join(current_path, "data", "test_PET")

    # # For each patient in folder moose_1
    # for patient_id in os.listdir(moose_path1):
    #     print("Patient: ", patient_id)
    #
    #     if patient_id != "MPC_249_20130306":
    #         continue
    #
    #
    #     try:
    #         # Get label path
    #         label_folder = "labels/sim_space/similarity-space"
    #         patient_path = os.path.join(moose_path1, patient_id)
    #         # label_path = os.path.join(patient_path, "MOOSE-" + patient_id, label_folder, "Spine.nii.gz")
    #         label_path = os.path.join(patient_path, "Bones_CT.nii.gz")
    #
    #         # Load label
    #
    #
    #         # Load label and perform the shaping
    #         mask = nib.load(label_path)
    #         mask = binarize(mask, 41)
    #         segmentation = copy.deepcopy(mask)
    #         spine_filled = fill_spinal_holes(segmentation, 3, 3)
    #         segmentation = copy.deepcopy(mask)
    #         spine_dilated = dilate_spine(segmentation, 3, True)
    #         segmentation = copy.deepcopy(mask)
    #         spine_cylinder = spine_as_cylinder(segmentation, 3)
    #         segmentation = copy.deepcopy(mask)
    #
    #         # Save cropped PET
    #         save_dir = os.path.join(save_path, patient_id)
    #         nib.save(segmentation, save_dir + "/mask_CT_original.nii.gz")
    #         nib.save(spine_filled, save_dir + "/mask_CT_fill_holes.nii.gz")
    #         nib.save(spine_dilated, save_dir + "/mask_CT_dilation.nii.gz")
    #         nib.save(spine_cylinder, save_dir + "/mask_CT_cylinder.nii.gz")
    #         print("Saved: ", patient_id)
    #
    #     except FileNotFoundError:
    #         print("FileNotFoundError for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/move_segm_log.txt", "a") as file:
    #             file.write(f"{patient_id}: FileNotFoundError\n")
    #         continue
    #
    #     except StopIteration:
    #         print("StopIteration for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/move_segm_log.txt", "a") as file:
    #             file.write(f"{patient_id}: StopIteration\n")
    #         continue
    #
    #     except Exception as e:
    #         print(f"Unknown error ({e}) for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/move_segm_log.txt", "a") as file:
    #             file.write(f"{patient_id}: Unknown error ({e})\n")
    #         continue
    #
    # # For each patient in folder moose_2
    # for patient_id in os.listdir(moose_path2):
    #
    #     if patient_id != "MPC_249_20130306":
    #         continue
    #
    #     print("Patient: ", patient_id)
    #
    #     try:
    #
    #         # # Find the label folder for moose 2.0
    #         start_seq = "moosez"
    #         label_folder = [d for d in os.listdir(patient_path) if d.startswith(start_seq)]
    #         label_path = os.path.join(patient_path, label_folder[0], "segmentations", "CT_Bones_V1_CT_0000.nii.gz")
    #         # label_path = os.path.join(patient_path, "Bones.nii.gz")
    #
    #         # Load label
    #         mask = nib.load(label_path)
    #
    #         # Perform the shaping
    #         mask = nib.load(label_path)
    #         mask = binarize(mask, 15)
    #         spine_filled = fill_spinal_holes(mask, 3, 3)
    #         mask = nib.load(label_path)
    #         mask = binarize(mask, 15)
    #         spine_dilated = dilate_spine(mask, 3, True)
    #         mask = nib.load(label_path)
    #         mask = binarize(mask, 15)
    #         spine_cylinder = spine_as_cylinder(mask, 3)
    #         mask = nib.load(label_path)
    #         mask = binarize(mask, 15)
    #
    #         # Save cropped PET
    #         save_dir = os.path.join(save_path, patient_id)
    #         nib.save(mask, save_dir + "/mask_CT_original.nii.gz")
    #         nib.save(spine_filled, save_dir + "/mask_CT_fill_holes.nii.gz")
    #         nib.save(spine_dilated, save_dir + "/mask_CT_dilation.nii.gz")
    #         nib.save(spine_cylinder, save_dir + "/mask_CT_cylinder.nii.gz")
    #         print("Saved: ", patient_id)
    #
    #     except FileNotFoundError:
    #         print("FileNotFoundError for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/move_segm_log.txt", "a") as file:
    #             file.write(f"{patient_id}: FileNotFoundError\n")
    #
    #     except StopIteration:
    #         print("StopIteration for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/move_segm_log.txt", "a") as file:
    #             file.write(f"{patient_id}: StopIteration\n")
    #
    #     except Exception as e:
    #         print(f"Unknown error ({e}) for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/move_segm_log.txt", "a") as file:
    #             file.write(f"{patient_id}: Unknown error ({e})\n")

    # Healthy patients cropping

    moose_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Healthy/moose_output"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Healthy/HEALTHY-PET-CT/FDG-PET-CT-Lesions"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/healthy_patients"

    next_start = "PETCT_9a66a81ad1"
    patients = os.listdir(moose_path)
    index = patients.index(next_start)

    # For each patient in folder moose_2
    for patient_id in patients:
        print("Patient: ", patient_id)

        try:

            # Find the label folder for moose on healthy patients
            patient_label_path = os.path.join(moose_path, patient_id)
            patient_moose_dir = [d for d in os.listdir(patient_label_path) if d.startswith("moosez")]
            label_folder = os.path.join(patient_label_path, patient_moose_dir[0], "segmentations")
            label_name = [d for d in os.listdir(label_folder) if ".nii.gz" in d]
            label_path = os.path.join(label_folder, label_name[0])

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
            with open(current_path + "/log/healthy_cropping_CT.txt", "a") as file:
                file.write(f"{patient_id}: FileNotFoundError\n")

        except StopIteration:
            print("StopIteration for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/healthy_cropping_CT.txt", "a") as file:
                file.write(f"{patient_id}: StopIteration\n")

        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/healthy_cropping_CT.txt", "a") as file:
                file.write(f"{patient_id}: Unknown error ({e})\n")

import os
import nibabel as nib
import numpy as np
import pandas as pd
from segmentation_processing.crop_spine_from_PET import crop_spine_shape


if __name__ == "__main__":

    # All patients path
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    moose_path1 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_1"
    moose_path2 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_2"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/sick_patients"
    # save_path = os.path.join(current_path, "data", "test_PET")

    # PET cropping functions
    shapes = ["original", "fill_holes", "dilation", "cylinder"]

    # # Define an iteration index and loop limit
    # i = 0
    # max_loops = 4
    # checkpoint = False

    # # For each patient in folder moose_1
    # for patient_id in os.listdir(moose_path1):
    #     # Condition to execute just for patients not already done
    #     if os.path.isdir(os.path.join(save_path, patient_id)):
    #         continue
    #     print("Patient: ", patient_id)
    #
    #     # # Continue from where it stopped
    #     # if not ((patient_id == "MPC_206_20160330") | checkpoint):
    #     #     continue
    #     # checkpoint = True
    #
    #     try:
    #         # Get label and PET path
    #         label_folder = "labels/sim_space/similarity-space"
    #         patient_path = os.path.join(moose_path1, patient_id)
    #         # label_path = os.path.join(patient_path, "MOOSE-" + patient_id, label_folder, "Spine.nii.gz")
    #         label_path = os.path.join(patient_path, "Bones.nii.gz")
    #         pet_path = os.path.join(data_path, patient_id, "PT.nii")
    #
    #         # Shape the PET image
    #         for function in shapes:
    #
    #             # Load label and PET
    #             segmentation_file = nib.load(label_path)
    #             pet_file = nib.load(pet_path)
    #
    #             # Perform the cropping
    #             cut_pet = crop_spine_shape(input_nifti=pet_file,
    #                                        mask=segmentation_file,
    #                                        shape=function,
    #                                        segmentation_value=41)
    #
    #             # Save cropped PET
    #             save_dir = os.path.join(save_path, patient_id)
    #             os.makedirs(save_dir, exist_ok=True)
    #             nib.save(cut_pet, save_dir + f"/PT_{function}.nii.gz")
    #             print("Saved: ", "PT_" + function + ".nii.gz")
    #
    #         # Limit the loops
    #         # if i == (max_loops - 1):
    #         #     break
    #         # i += 1
    #
    #     except FileNotFoundError:
    #         print("FileNotFoundError for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/log_moose1.txt", "a") as file:
    #             file.write(f"{patient_id}: FileNotFoundError\n")
    #         continue
    #
    #     except StopIteration:
    #         print("StopIteration for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/log_moose1.txt", "a") as file:
    #             file.write(f"{patient_id}: StopIteration\n")
    #         continue
    #
    #     except Exception as e:
    #         print(f"Unknown error ({e}) for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/log_moose1.txt", "a") as file:
    #             file.write(f"{patient_id}: Unknown error ({e})\n")
    #         continue
    #
    # # For each patient in folder moose_2
    # for patient_id in os.listdir(moose_path2):
    #     # Condition to execute just for patients not already done
    #     if os.path.isdir(os.path.join(save_path, patient_id)):
    #         continue
    #     print("Patient: ", patient_id)
    #
    #     try:
    #
    #         # Get PET and segmentation path
    #         patient_path = os.path.join(moose_path2, patient_id)
    #         pet_path = os.path.join(data_path, patient_id, "PT.nii")
    #
    #         # # Find the label folder for moose 2.0
    #         # start_seq = "moosez"
    #         # label_folder = [d for d in os.listdir(patient_path) if d.startswith(start_seq)]
    #         # label_path = os.path.join(patient_path, label_folder[0], "segmentations", "CT_Bones_V1_CT_0000.nii.gz")
    #         label_path = os.path.join(patient_path, "Bones.nii.gz")
    #
    #         # Shape the PET image
    #         for function in shapes:
    #
    #             # Load label
    #             segmentation_file = nib.load(label_path)
    #             pet_file = nib.load(pet_path)
    #
    #             # Perform the cropping
    #             cut_pet = crop_spine_shape(input_nifti=pet_file,
    #                                        mask=segmentation_file,
    #                                        shape=function,
    #                                        segmentation_value=15)
    #
    #             # Saved cropped PET
    #             save_dir = os.path.join(save_path, patient_id)
    #             os.makedirs(save_dir, exist_ok=True)
    #             nib.save(cut_pet, save_dir + f"/PT_{function}.nii.gz")
    #             print("Saved: ", f"PT_{function}.nii.gz")
    #
    #         # # Limit the loops
    #         # if i == max_loops:
    #         #     break
    #         # i += 1
    #
    #     except FileNotFoundError:
    #         print("FileNotFoundError for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/cropping_report.txt", "a") as file:
    #             file.write(f"{patient_id}: FileNotFoundError\n")
    #
    #     except StopIteration:
    #         print("StopIteration for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/cropping_report.txt", "a") as file:
    #             file.write(f"{patient_id}: StopIteration\n")
    #
    #     except Exception as e:
    #         print(f"Unknown error ({e}) for patient: ", patient_id)
    #         # Write the patient id which had an error
    #         with open(current_path + "/log/cropping_report.txt", "a") as file:
    #             file.write(f"{patient_id}: Unknown error ({e})\n")

    # Healthy patients cropping

    moose_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Healthy/moose_output"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Healthy/HEALTHY-PET-CT/FDG-PET-CT-Lesions"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/healthy_patients"

    # declare dataframe with the cropped image max values
    columns = ['patient_id', 'shape', 'max_value']
    max_values = pd.DataFrame(columns=columns)

    # For each patient in folder moose_2
    for patient_id in os.listdir(moose_path):
        # Condition to execute just for patients not already done
        # if os.path.isdir(os.path.join(save_path, patient_id)):
        #     continue
        print("Patient: ", patient_id)

        try:

            # Get PET path for healthy patients
            patient_pet_path = os.path.join(data_path, patient_id)
            pet_name = [d for d in os.listdir(patient_pet_path) if d.endswith(".nii")]
            pet_path = os.path.join(patient_pet_path, pet_name[0])

            # Find the label folder for moose on healthy patients
            patient_label_path = os.path.join(moose_path, patient_id)
            label_folder = [d for d in os.listdir(patient_label_path) if d.startswith("moosez")]
            label_folder = os.path.join(patient_label_path, label_folder[0], "segmentations")
            label_name = [d for d in os.listdir(label_folder) if ".nii.gz" in d]
            label_path = os.path.join(label_folder, label_name[0])

            # Shape the PET image
            for function in shapes:

                # Load label
                segmentation_file = nib.load(label_path)
                pet_file = nib.load(pet_path)

                # Perform the cropping
                cut_pet = crop_spine_shape(input_nifti=pet_file,
                                           mask=segmentation_file,
                                           shape=function,
                                           segmentation_value=15)

                # Get crop PET max voxel value
                pet_array = cut_pet.get_fdata()
                max_value = np.amax(pet_array)
                print(f"Max value: {max_value}")
                if max_value < 20:
                    new_line = [patient_id, function, max_value]
                    max_values.loc[len(max_values)] = new_line
                    with open(current_path + "/log/cropped_img_max_val.txt", "a") as file:
                        file.write(f"{patient_id} with function '{function}' has max value: {max_value}\n")

                # Saved cropped PET
                save_dir = os.path.join(save_path, patient_id)
                os.makedirs(save_dir, exist_ok=True)
                nib.save(cut_pet, save_dir + f"/PT_{function}.nii")
                print("Saved:", f"PT_{function}.nii")

            # # Limit the loops
            # if i == max_loops:
            #     break
            # i += 1

        except FileNotFoundError:
            print("FileNotFoundError for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/healthy_cropping_report.txt", "a") as file:
                file.write(f"{patient_id}: FileNotFoundError\n")

        except StopIteration:
            print("StopIteration for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/healthy_cropping_report.txt", "a") as file:
                file.write(f"{patient_id}: StopIteration\n")

        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/healthy_cropping_report.txt", "a") as file:
                file.write(f"{patient_id}: Unknown error ({e})\n")

    max_values.to_csv('log/dataframe_max_val.csv', index=True)

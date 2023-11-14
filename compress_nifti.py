import os
import nibabel as nib

if __name__ == "__main__":

    # All patients path
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    moose_path1 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_1"
    moose_path2 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_2"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"
    save_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/sick_patients"
    # save_path = os.path.join(current_path, "data", "test_PET")

    # For each patient in folder moose_1
    for patient_id in os.listdir(save_path):
        print("Patient: ", patient_id)
        try:
            dir_sick_patient = os.path.join(save_path, patient_id)
            if os.path.isdir(dir_sick_patient):
                nifti_files = [file_name for file_name in os.listdir(dir_sick_patient) if file_name.endswith(".nii")]
                for file in nifti_files:
                    file_path = os.path.join(dir_sick_patient, file)
                    print("\nPath towards the NIfTI file:  ", file_path)
                    nifti = nib.load(file_path)
                    nib.save(nifti, file_path + ".gz")
                    os.remove(file_path)


            else:
                nifti_files = "None"
            print(f"List of files to converted in the folder {patient_id}: {nifti_files}")

            break  # Da togliere!!!

        except FileNotFoundError:
            print("FileNotFoundError for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/compression_log.txt", "a") as file:
                file.write(f"{patient_id}: FileNotFoundError\n")
            continue

        except StopIteration:
            print("StopIteration for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/compression_log.txt", "a") as file:
                file.write(f"{patient_id}: StopIteration\n")
            continue

        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/compression_log.txt", "a") as file:
                file.write(f"{patient_id}: Unknown error ({e})\n")
            continue


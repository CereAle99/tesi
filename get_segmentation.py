import nibabel as nib
import os

if __name__ == "__main__":

    # All patients path
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
    moose_path1 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_1"
    moose_path2 = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/Moose_output/moose_2"
    data_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"
    sick_patients_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/sick_patients"
    healthy_patients_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/healthy_patients"

    # Checkpoint last execution
    next_start = "MPC_149_20060705"
    patients = os.listdir(sick_patients_path)
    index = patients.index("MPC_149_20060705")

    # For each patient in folder sick_patients
    for patient_id in patients:
        print("Patient: ", patient_id)

        if patient_id != "MPC_224_20160118":
            continue

        try:
            sick_patient = os.path.join(sick_patients_path, patient_id)
            if os.path.isdir(sick_patient):
                nifti_files = [file_name for file_name in os.listdir(sick_patient) if file_name.startswith("PT")]
                print(f"List of files in the folder {patient_id}: {nifti_files}")
                for file in nifti_files:
                    file_path = os.path.join(sick_patient, file)
                    print("\nName of the NIfTI file:  ", file)
                    nib_nifti = nib.load(file_path)
                    print("load")
                    image = nib_nifti.get_fdata()
                    image[image != 0] = 1
                    image[image == 0] = 0
                    segmentation = nib.Nifti1Image(image, nib_nifti.affine, nib_nifti.header)
                    nib.save(segmentation, os.path.join(sick_patient, "mask_" + file))
                    print("save")

        except FileNotFoundError:
            print("FileNotFoundError for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/find_segmentation.txt", "a") as file:
                file.write(f"{patient_id}: FileNotFoundError\n")
            continue

        except StopIteration:
            print("StopIteration for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/find_segmentation.txt", "a") as file:
                file.write(f"{patient_id}: StopIteration\n")
            continue

        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/log/find_segmentation.txt", "a") as file:
                file.write(f"{patient_id}: Unknown error ({e})\n")
            continue



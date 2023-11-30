import pandas as pd
import radiomics
import SimpleITK as sitk
import os
import six
from radiomics import featureextractor
import logging
import csv


if __name__ == "__main__":

    # set pyradiomics verbose to False
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    # Get current path and all useful paths
    current_path = os.getcwd()
    print(current_path)
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    segmentations_dir = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/"
    sick_patients_path = segmentations_dir + "sick_patients"
    healthy_patients_path = segmentations_dir + "healthy_patients"

    # Create the dataframe where to store all the features
    # feats_dataframe = pd.DataFrame()

    fieldnames = None

    # Checkpoint last execution
    next_start = "MPC_2710_20070302"
    patients = os.listdir(sick_patients_path)
    index = patients.index(next_start)

    for patient_id in patients[index:]:
        print("Patient: ", patient_id)

        if patient_id == "MPC_145_20081110":
            continue

        try:
            # Select the pet images in patient folder
            pet_files = [pet for pet in os.listdir(os.path.join(sick_patients_path, patient_id)) if pet.startswith("PT_or")]

            for pet_file in pet_files:
                print(pet_file)
                
                # Load the image and the segmentation
                image_path = os.path.join(sick_patients_path, patient_id, pet_file)
                imageName = sitk.ReadImage(image_path)
                mask_path = os.path.join(sick_patients_path, patient_id, f"mask_{pet_file}")
                maskName = sitk.ReadImage(mask_path)

                # Set the features extractor
                extr_params = os.path.join(current_path, "feat_extr_test.yaml")
                extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(extr_params)

                # Add new line in features dataset
                new_line = {"patient_id": patient_id, "mask_shape": pet_file[3:-7]}

                # Start the feature extraction
                result = extractor.execute(imageName, maskName)
                for key, val in six.iteritems(result):
                    # Add feature value to the features dataset
                    new_line[key] = val
                    # print(f"Feature {key}: {val}")

                with open(os.path.join(sick_patients_path, "PT_rad_feats.csv"), "a", newline="") as csvfile:

                    # create writer
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # Add dict new line on the file
                    writer.writerow(new_line)

                # Add patient features line to the features dataframe
                # feats_dataframe = feats_dataframe._append(new_line, ignore_index=True)
                print(f"Patient {patient_id} features stored in dataframe\n")
                # result = extractor.execute(imageName, maskName, voxelBased=True)

        except Exception as e:
            print(f"Unknown error ({e}) for patient: ", patient_id)
            # Write the patient id which had an error
            with open(current_path + "/feat_extr_log.txt", "a") as file:
                file.write(f"{patient_id}:  ({e})\n")

    # Features dataframe save
    # feats_dataframe.to_csv(os.path.join(sick_patients_path, "PT_rad_feats.csv"))


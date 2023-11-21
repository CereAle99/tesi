import pandas as pd
import radiomics
import SimpleITK as sitk
import os
import six
from radiomics import featureextractor
import logging


if __name__ == "__main__":

    # set pyradiomics verbose to False
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    # Get current path and all useful paths
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
    segmentations_dir = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/"
    sick_patients_path = segmentations_dir + "sick_patients"
    healthy_patients_path = segmentations_dir + "healthy_patients"

    # Create the dataframe where to store all the features
    feats_dataframe = pd.DataFrame()

    for patient_id in os.listdir(sick_patients_path):
        print("Patient: ", patient_id)

        # Select the pet images in patient folder
        pet_files = [pet for pet in os.listdir(os.path.join(sick_patients_path, patient_id)) if pet.startswith("PT")]

        for pet_file in pet_files:

            # Load the image and the segmentation
            image_path = os.path.join(sick_patients_path, patient_id, pet_file)
            imageName = sitk.ReadImage(image_path)
            mask_path = os.path.join(sick_patients_path, patient_id, f"mask_{pet_file}")
            maskName = sitk.ReadImage(mask_path)

            # Set the features extractor
            extr_params = os.path.join(current_path, "parameters", "feat_extr_test.yaml")
            extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(extr_params)

            # Add new line in features dataset
            new_line = {"patient_id": patient_id, "mask_shape": pet_file[3:-7]}

            # Start the feature extraction
            result = extractor.execute(imageName, maskName)
            for key, val in six.iteritems(result):
                # Add feature value to the features dataset
                new_line[key] = val
                # print(f"Feature {key}: {val}")

            # Add patient features line to the features dataframe
            feats_dataframe = feats_dataframe._append(new_line, ignore_index=True)
            print(f"Patient {patient_id} features stored in dataframe\n")
            # result = extractor.execute(imageName, maskName, voxelBased=True)

    # Features dataframe save
    feats_dataframe.to_csv(os.path.join(current_path, "data", "PT_rad_feats.csv"))

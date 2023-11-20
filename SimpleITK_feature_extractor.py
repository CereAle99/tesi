import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import os
import six
from radiomics import featureextractor, getTestCase


if __name__ == "__main__":

    # Get current path and all useful paths
    current_path = os.getcwd()
    shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
    segmentations_dir = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/"
    sick_patients_path = segmentations_dir + "sick_patients"
    healthy_patients_path = segmentations_dir + "healthy_patients"

    # Create the dataframe where to store all the features

    features = pd.DataFrame()

    for patient_id in os.listdir(sick_patients_path):
        print("Patient: ", patient_id)

        # Select the pet images in patient folder
        pet_files = [pet for pet in os.path.join(sick_patients_path, patient_id) if pet.startswith("PT")]

        for pet_file in pet_files:

            # Load the image and the segmentation
            image_path = os.path.join(sick_patients_path, patient_id, pet_file)
            imageName = sitk.ReadImage(image_path)
            mask_path = os.path.join(sick_patients_path, patient_id, f"mask_{pet_file}")
            maskName = sitk.ReadImage(mask_path)

            # Set the features extractor
            extr_params = os.path.join(current_path, "parameters", "feat_extr_test.yaml")
            extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(extr_params)

            new_line = {"patient_id": patient_id, "mask_shape": pet_file[3:-7]}
            result = extractor.execute(imageName, maskName)
            for key, val in six.iteritems(result):
                new_line[key] = val
                print(f"Feature {key}: {val}")
            features = features.append(new_line, ignore_index=True)
            print("Stored features in ")
            break
            # result = extractor.execute(imageName, maskName, voxelBased=True)
        break

    features.to_csv(os.path.join(current_path, "data", "PT_rad_feats.csv"))




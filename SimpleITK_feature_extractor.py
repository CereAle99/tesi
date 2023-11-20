import numpy as np
import radiomics
import SimpleITK as sitk
import os
import six
from radiomics import featureextractor, getTestCase


if __name__ == "__main__":

    # Get current path
    current_path = os.getcwd()

    # Load the image
    image_path = os.path.join(current_path, "data", "path_all_tua_immagine.nii")
    imageName = sitk.ReadImage(image_path)
    mask_path = os.path.join(current_path, "data", "path_all_tua_immagine.nii")
    maskName = sitk.ReadImage(mask_path)

    # Set the features extractor
    extr_params = os.path.join(current_path, "parameters", "feat_extr_test.yaml")
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(extr_params)

    result = extractor.execute(imageName, maskName)
    for key, val in six.iteritems(result):
        print("\t%s: %s" % (key, val))

    result = extractor.execute(imageName, maskName, voxelBased=True)
    for key, val in six.iteritems(result):
        if isinstance(val, sitk.Image):  # Feature map
            sitk.WriteImage(val, key + '.nrrd', True)
            print("Stored feature %s in %s" % (key, key + ".nrrd"))
        else:  # Diagnostic information
            print("\t%s: %s" % (key, val))




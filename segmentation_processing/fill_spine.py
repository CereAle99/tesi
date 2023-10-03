import nibabel as nib
import numpy as np
import os

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

img = nib.load(data_path + "Spine.nii.gz")
data = img.get_fdata()



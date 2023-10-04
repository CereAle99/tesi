import nibabel as nib
from scipy import ndimage
import numpy as np
import os
from close_spine_area import dilate_spine

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the NIfTI segmentation and put it into a numpy array
img = nib.load(data_path + "Spine.nii.gz")
image = img.get_fdata()
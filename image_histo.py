import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Data paths
shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
data_directory = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"

# Image info to seek the path
patient_id = "MPC_264_20160216"
file_name = "PT.nii"

# Load the image
file_path = os.path.join(shared_dir_path, data_directory, patient_id, file_name)
nifti_img = nib.load(file_path)

# Get the image array
ct_data = nifti_img.get_fdata()

# Shape the array as 1D
flat_data = ct_data.flatten()

# Draw the histogram
plt.hist(flat_data, bins=100, color='blue', edgecolor='black')
plt.yscale('log')  # Imposta la scala logaritmica sull'asse y
plt.title('Grey levels histogram')
plt.xlabel('Grey levels')
plt.ylabel('Frequency')
plt.show()

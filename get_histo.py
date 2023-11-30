import nibabel as nib
import os
from segmentation_processing.image_histo import image_histo


if __name__ == "__main__":

    # Data paths
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    data_directory = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Original/PET-CT"

    # Image info to seek the path
    patient_id = "MPC_264_20160216"
    file_name = "PT.nii"

    # Load the image
    file_path = os.path.join(shared_dir_path, data_directory, patient_id, file_name)
    nifti_img = nib.load(file_path)

    image_histo(nifti_img)




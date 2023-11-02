import os
import subprocess

# All patients path
current_path = os.getcwd()
shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
patients_folder = "/Genomed4All_Data/MultipleMieloma/HEALTHY-PET-CT/FDG-PET-CT-Lesions"
dicom_path = shared_dir_path + patients_folder
# print(os.listdir(dicom_path)[0])

checkpoint = -1

# Loop for patients folders
for num, patient_id in enumerate(os.listdir(dicom_path)):

    if num <= checkpoint:
        continue

    print(patient_id)
    patient_path = os.path.join(dicom_path, patient_id)
    images_path = os.path.join(patient_path, os.listdir(patient_path)[0])  # may have more sets of images (?)

    # Imaging name dict
    output_filename = {0: "CT.nii",
                       1: "PET.nii",
                       2: "Segmentation.nii.gz"}

    # Loop for CT, PET and segmentation folders
    for i, imaging_type in enumerate(os.listdir(images_path)):

        print(os.listdir(images_path)[i])

        # Define the folder with DICOM files to convert
        input_folder = os.path.join(images_path, imaging_type)

        # Command to convert the DICOM files in the folder
        dcm2niix = "/home/gatvmatteo/anaconda3/pkgs/dcm2niix-1.0.20230411-h00ab1b0_0/bin/dcm2niix"
        command = [dcm2niix, "-o", patient_path, input_folder]

        # Run the command
        subprocess.run(command)

    print(f"Conversion completed. Patient nº{num}: {patient_id}")

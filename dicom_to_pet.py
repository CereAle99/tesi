import os
import subprocess

# All patients path
current_path = os.getcwd()
shared_dir_path = "/run/user/1000/gvfs/afp-volume:host=RackStation.local,user=aceresi,volume=Genomed"
patients_folder = "/Genomed4All_Data/MultipleMieloma/Healthy/HEALTHY-PET-CT/FDG-PET-CT-Lesions/"
dicom_path = shared_dir_path + patients_folder
# print(os.listdir(dicom_path)[0])

# Last file completely converted
checkpoint = -1
single_pet = "PETCT_ca16242e89"

# Loop for patients folders

for num, patient_id in enumerate(os.listdir(dicom_path)):
    print(num)

    if (num <= checkpoint) | (not patient_id == single_pet):
        continue

    print(patient_id)
    patient_path = os.path.join(dicom_path, patient_id)
    imageset_list = [directory for directory in os.listdir(dicom_path)
                     if os.path.isdir(os.path.join(dicom_path, directory))]
    images_path = os.path.join(patient_path, imageset_list[1])  # "1" because may have more sets of images (?)

    # Imaging name dict
    output_filename = {0: "CT.nii",
                       1: "PET.nii",
                       2: "Segmentation.nii.gz"}

    # Loop for CT, PET and segmentation folders
    for i, imaging_type in enumerate(os.listdir(images_path)):

        if "PET" not in imaging_type:
            continue

        print(os.listdir(images_path)[i])

        # Define the folder with DICOM files to convert
        input_folder = os.path.join(images_path, imaging_type)

        # Command to convert the DICOM files in the folder
        dcm2niix = "/home/cronos/anaconda3/envs/medical_images/bin/dcm2niix"
        command = [dcm2niix, "-o", patient_path, input_folder]

        # Run the command
        subprocess.run(command)

    print(f"\nConversion completed. Patient nº{num}: {patient_id}\n")

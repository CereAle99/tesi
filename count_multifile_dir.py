import os
import csv


if __name__ == "__main__":
    
    # Get all useful paths
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    segmentations_dir = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/"
    healthy_patients_path = segmentations_dir + "healthy_patients"

    # Get patients folders list
    patients = os.listdir(healthy_patients_path)

    # List of patients with more than 1 set of images
    ambiguous_segm = []

    print("Patients with more than 1 set of images:\n")

    for patient in patients:
        
        # list of dirs in patient folder
        patient_dirs = [dir for dir in os.listdir(os.path.join(healthy_patients_path, patient)) if os.path.isdir(dir)]

        # add to the list the patient id if patient_dirs is > 1
        if len(patient_dirs) > 1:
            
            ambiguous_segm.append(patient)
            print(patient)

    # Save the patient id list
    with open("healthy_ambiguous_segm.txt", "a", newline="") as csv_file:

        # create writer
        csv_writer = csv.writer(csv_file)

        # Add list to csv file
        csv_writer.writerow(ambiguous_segm)
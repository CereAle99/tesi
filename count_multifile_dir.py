import os
import csv


if __name__ == "__main__":
    
    # Get all useful paths
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    healthy_patients_path = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/Healthy/HEALTHY-PET-CT/FDG-PET-CT-Lesions"

    # Get patients folders list
    patients = os.listdir(healthy_patients_path)

    # List of patients with more than 1 set of images
    ambiguous_segm = []

    print("Patients with more than 1 set of images:\n")

    for patient in patients:
        
        # list of dirs in patient folder
        patient_dir = os.path.join(healthy_patients_path, patient)
        image_sets = [dir for dir in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, dir))]
        print(image_sets)

        # add to the list the patient id if patient_dirs is > 1
        if len(image_sets) > 1:
            
            ambiguous_segm.append(patient)
            print(patient)

    # Save the patient id list
    with open("healthy_ambiguous_segm.csv", "a", newline="") as csv_file:

        # create writer
        csv_writer = csv.writer(csv_file)

        # Add list to csv file
        csv_writer.writerow(ambiguous_segm)
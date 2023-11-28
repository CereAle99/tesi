import os


if __name__ == "__main__":
    
    # Get all useful paths
    shared_dir_path = "/run/user/1000/gvfs/smb-share:server=192.168.0.6,share=genomed"
    segmentations_dir = shared_dir_path + "/Genomed4All_Data/MultipleMieloma/spine_PET/"
    sick_patients_path = segmentations_dir + "sick_patients"

    # # Checkpoint last execution
    next_start = "MPC_2710_20070302"
    patients = [patient for patient in os.listdir(sick_patients_path) if patient.startswith("MPC")]
    index = patients.index(next_start)

    # Loop for all the patients
    for patient_id in patients:
        print(patient_id, ".  Pregresses: ", patients.index(patient_id)/len(patients) * 100)
        patient_dir = os.path.join(sick_patients_path, patient_id)
        
        # List of paths towards files to remove with a specific start
        files_toremove = [file for file in os.listdir(patient_dir) if file.startswith("~g")]
        
        # Loop through all the files to remove
        for file in files_toremove:
            # Remove the files
            os.remove(os.path.join(patient_dir, file))
            print("removed")
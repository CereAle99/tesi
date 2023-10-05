import numpy as np
import nibabel as nib
import os

if __name__=="__main__":
    # All patients path
    path = "data"

    for folder in os.listdir(path):
        try:
            print("Patient: ", folder)
            # Get label path
            patient_path = os.path.join(path,folder)
            image_path = os.path.join(patient_path, "PET/PT.nii")

            # Load label
            img_obj = nib.load(image_path)
            img = img_obj.get_fdata()
            # Transform label
            img = np.resize(img, [512, 512, 239])
            img_resized = nib.Nifti1Image(img, img_obj.affine, img_obj.header)

            # Save label
            save_path = os.path.join(patient_path, "PET-Resized")
            os.makedirs(save_path, exist_ok=True)
            if not os.path.isfile(save_path+'/PT.nii'):
                print("Saving PET")
                nib.save(img_resized, save_path+'/PT.nii')
        except: pass




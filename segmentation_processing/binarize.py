import nibabel as nib
import os

# Get the present directory path and data directory
current_directory = os.getcwd()
data_path = current_directory + '/data/'

# Load the segmentation NIfTI file
spine = nib.load(data_path + "CT_Bones_V1_CT_0000.nii.gz")


def binarize(input_nifti, label):

    # Load the nifti file image
    image = input_nifti.get_fdata()

    # Binarize the image for the label value
    if label == 1:
        pass
    elif label:
        image[image != label] = 0
        image[image == label] = 1

    # Save the image in nifti file
    binarized_nifti = nib.Nifti1Image(image, input_nifti.affine, input_nifti.header)

    return binarized_nifti


# Binarize segmentation
final_image = binarize(spine, 15)

# Save the modified segmentation as a NIfTI file
nib.save(final_image, os.path.join(data_path, 'binarized_bone.nii.gz'))

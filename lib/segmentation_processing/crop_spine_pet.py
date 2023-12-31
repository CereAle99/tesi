import nibabel as nib
from lib.segmentation_processing.resize_pet import pet_compatible_to_ct
from lib.segmentation_processing.segm_fill_spine import fill_spinal_holes
from lib.segmentation_processing.segm_dilation import dilate_spine
from lib.segmentation_processing.segm_cylinder import spine_as_cylinder
from lib.segmentation_processing.segm_binarize import binarize


def crop_spine_shape(input_nifti, mask, shape="original", segmentation_value=41):
    """

    Args:
        input_nifti:
        mask:
        shape:
        segmentation_value:

    Returns:

    """

    mask = binarize(mask, segmentation_value)

    # Apply shape function on segmentation
    if shape == "fill_holes":
        print(shape)
        mask = fill_spinal_holes(mask, 3, 3)
    elif shape == "dilation":
        print(shape)
        mask = dilate_spine(mask, 3, True)
    elif shape == "cylinder":
        print(shape)
        mask = spine_as_cylinder(mask, 3)
    elif shape == "original":
        print(shape)
    else:
        print("Shape invalid. Going with the original shape.")
    print("done shaping")

    # Make PET image and spine segmentation image compatibles
    resized_pet, resized_mask = pet_compatible_to_ct(input_nifti, mask)
    print("done resizing")

    # Put the segmentation into a numpy array
    segmentation = resized_mask.get_fdata()

    # Put the image into a numpy array
    image = resized_pet.get_fdata()

    # Cut the PET image
    cut_image = image * segmentation
    print(f"done cutting")

    # Save cut image in a NIfTI file
    cut_file = nib.Nifti1Image(cut_image, resized_pet.affine, resized_pet.header)

    return cut_file

import nibabel as nib
from segmentation_processing.resizePET_Ale import pet_compatible_to_ct
from segmentation_processing.fill_spine import fill_spinal_holes
from segmentation_processing.close_spine_area import dilate_spine
from segmentation_processing.spine_outofthe_cylinder import spine_as_cylinder
from segmentation_processing.binarize import binarize


def crop_spine_shape(input_nifti, mask, shape="original", segmentation_value=41):
    """

    Args:
        input_nifti:
        mask:
        shape:
        segmentation_value:

    Returns:

    """

    # Apply shape function on segmentation
    if shape == "fill_holes":
        print(shape)
        mask = binarize(mask, segmentation_value)
        mask = fill_spinal_holes(mask, 3, 3)
    elif shape == "dilation":
        print(shape)
        mask = binarize(mask, segmentation_value)
        mask = dilate_spine(mask, 3, True)
    elif shape == "cylinder":
        print(shape)
        mask = binarize(mask, segmentation_value)
        mask = spine_as_cylinder(mask, 3)
    elif shape == "original":
        print(shape)
        mask = binarize(mask, segmentation_value)
    else:
        print("Shape invalid. Going with the original shape.")
        mask = binarize(mask, segmentation_value)
    print("done shaping")
    # Make PET image and spine segmentation image compatibles
    resized_pet, resized_mask = pet_compatible_to_ct(input_nifti, mask)
    print("done resizing")
    # Put the segmentation into a numpy array
    spine_mask = resized_mask.get_fdata()

    # Put the image into a numpy array
    image = resized_pet.get_fdata()

    # Cut the PET image
    cut_image = image * spine_mask
    print("done cutting")
    # Save cut image in a NIfTI file
    cut_file = nib.Nifti1Image(cut_image, resized_pet.affine, resized_pet.header)

    return cut_file

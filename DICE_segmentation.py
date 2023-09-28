import SimpleITK as sitk
import numpy as np

# Read segmentations from NIfTI files
segmentation_auto = sitk.ReadImage('segmentation_auto.nii.gz')
segmentation_ground_truth = sitk.ReadImage('segmentation_ground_truth.nii.gz')

# Extract image data as NumPy arrays
seg_auto_np = sitk.GetArrayFromImage(segmentation_auto)
seg_gt_np = sitk.GetArrayFromImage(segmentation_ground_truth)

# Calculate the numerator (intersection)
intersection = np.logical_and(seg_auto_np, seg_gt_np).sum()

# Calculate the denominator (sum of both segmentations)
union = seg_auto_np.sum() + seg_gt_np.sum()

# Calculate the DICE coefficient
dice_coefficient = 2.0 * intersection / union

print(f'The DICE coefficient is: {dice_coefficient}')

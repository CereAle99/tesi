import SimpleITK as sitk
import numpy as np
from skimage import measure

# Step 1: Load the segmented image
segmentation_image = sitk.ReadImage('segmentation.nii.gz')

# Step 2: Image preprocessing (optional)


# Step 3: Extract features
# Extract lesion volume features
label_statistics = sitk.LabelStatisticsImageFilter()
label_statistics.Execute(segmentation_image, segmentation_image)

volume = label_statistics.GetSum(1)  # Volume of the region labeled as 1 (lesions)

# Extract geometric features
# Calculate area and perimeter
label_map = sitk.LabelImageFilter().Execute(segmentation_image)
labels = np.unique(sitk.GetArrayFromImage(label_map))
for label in labels[1:]:  # Ignore the background label (0)
    region = (sitk.GetArrayFromImage(label_map) == label)
    area = np.sum(region)  # Calculate area
    contours = measure.find_contours(region, 0.5)
    perimeter = sum([len(contour) for contour in contours])  # Calculate perimeter
    # You can record or further process these geometric measures.

# Extract intensity statistics
intensity_statistics = sitk.StatisticsImageFilter()
intensity_statistics.Execute(segmentation_image)
mean_intensity = intensity_statistics.GetMean()  # Mean intensity within the region labeled as 1

# Extract texture features (e.g., contrast)
gray_image = sitk.GetArrayFromImage(segmentation_image)
contrast = sitk.GradientMagnitudeRecursiveGaussian(segmentation_image, sigma=1.0)
contrast_statistics = sitk.LabelStatisticsImageFilter()
contrast_statistics.Execute(contrast, label_map)
contrast_mean = contrast_statistics.GetMean(1)  # Mean contrast within the region labeled as 1

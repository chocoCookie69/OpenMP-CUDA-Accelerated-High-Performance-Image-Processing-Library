import cv2
import numpy as np
import image_processing

image = cv2.imread(r"E:\Program Files\Image Processing Library\ImageProcesingLibrary\ImageProcesingLibrary\data\Haardhik.jpg")
# Check if image was loaded properly
if image is None:
    raise ValueError("Image not found. Check the path.")

cv2.imshow('Original', image)
# Initialize processor
processor = image_processing.ImageProcessor()

# Apply invertCUDA
output = processor.invert_cuda(image)

# Show result

cv2.imshow('Inverted', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
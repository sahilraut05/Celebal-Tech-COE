import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('example.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Mean filter
mean_filter = cv2.blur(image, (5, 5))

# Apply Median filter
median_filter = cv2.medianBlur(image, 5)

# Apply Gaussian filter
gaussian_filter = cv2.GaussianBlur(image, (5, 5), 0)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(mean_filter)
plt.title('Mean Filter')
plt.subplot(1, 4, 3)
plt.imshow(median_filter)
plt.title('Median Filter')
plt.subplot(1, 4, 4)
plt.imshow(gaussian_filter)
plt.title('Gaussian Filter')
plt.show()

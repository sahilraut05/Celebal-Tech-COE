import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('example.jpg', 0)

# Contrast Stretching
min_val = np.min(image)
max_val = np.max(image)
contrast_stretched = ((image - min_val) / (max_val - min_val) * 255).astype('uint8')

# Histogram Equalization
hist_eq = cv2.equalizeHist(image)

# Image Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)

# Smoothing (Blurring)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 5, 2)
plt.imshow(contrast_stretched, cmap='gray')
plt.title('Contrast Stretching')
plt.subplot(1, 5, 3)
plt.imshow(hist_eq, cmap='gray')
plt.title('Histogram Equalization')
plt.subplot(1, 5, 4)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.subplot(1, 5, 5)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.show()

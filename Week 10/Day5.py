import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('example.jpg', 0)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a kernel
kernel = np.ones((5, 5), np.uint8)

# Dilation
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Erosion
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Opening
opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Closing
closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.subplot(1, 5, 2)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')
plt.subplot(1, 5, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.subplot(1, 5, 4)
plt.imshow(opening_image, cmap='gray')
plt.title('Opening Image')
plt.subplot(1, 5, 5)
plt.imshow(closing_image, cmap='gray')
plt.title('Closing Image')
plt.show()

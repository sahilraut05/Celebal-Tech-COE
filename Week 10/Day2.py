import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('example.jpg', 0)

# Apply Sobel edge detection
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)

# Apply Prewitt edge detection
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittx = cv2.filter2D(image, cv2.CV_64F, kernelx)
prewitty = cv2.filter2D(image, cv2.CV_64F, kernely)
prewitt = cv2.magnitude(prewittx, prewitty)

# Apply Canny edge detection
canny = cv2.Canny(image, 100, 200)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Edge Detection')
plt.subplot(1, 4, 3)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.subplot(1, 4, 4)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('example.jpg')
rows, cols = image.shape[:2]

# Translation
M_translation = np.float32([[1, 0, 50], [0, 1, 50]])
translated_image = cv2.warpAffine(image, M_translation, (cols, rows))

# Rotation
M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(image, M_rotation, (cols, rows))

# Scaling
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine_image = cv2.warpAffine(image, M_affine, (cols, rows))

# Perspective Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [200, 200]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
perspective_image = cv2.warpPerspective(image, M_perspective, (cols, rows))

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 5, 2)
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.title('Translated Image')
plt.subplot(1, 5, 3)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')
plt.subplot(1, 5, 4)
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title('Scaled Image')
plt.subplot(1, 5, 5)
plt.imshow(cv2.cvtColor(affine_image, cv2.COLOR_BGR2RGB))
plt.title('Affine Image')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image
image_path = 'example.jpg'

# Check if the image path exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image path '{image_path}' does not exist. Please provide a valid path.")

image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Failed to load image. Please check if the file at '{image_path}' is a valid image.")

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate the histogram for each color channel
channels = ('b', 'g', 'r')
hist_data = {}
for i, channel in enumerate(channels):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    hist_data[channel] = hist

# Plot histograms for each color channel
plt.figure(figsize=(14, 7))

for i, channel in enumerate(channels):
    plt.subplot(1, 3, i + 1)
    plt.plot(hist_data[channel], color=channel)
    plt.title(f'{channel.upper()} channel histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Advanced Visualization: 2D Histogram for color channels
plt.figure(figsize=(14, 7))

# Plotting 2D histogram for Red and Green channels
plt.subplot(1, 2, 1)
plt.hist2d(image_rgb[:, :, 0].ravel(), image_rgb[:, :, 1].ravel(), bins=32, range=[[0, 256], [0, 256]], cmap='Reds')
plt.colorbar()
plt.title('2D Histogram (Red vs Green)')
plt.xlabel('Red Intensity')
plt.ylabel('Green Intensity')

# Plotting 2D histogram for Green and Blue channels
plt.subplot(1, 2, 2)
plt.hist2d(image_rgb[:, :, 1].ravel(), image_rgb[:, :, 2].ravel(), bins=32, range=[[0, 256], [0, 256]], cmap='Greens')
plt.colorbar()
plt.title('2D Histogram (Green vs Blue)')
plt.xlabel('Green Intensity')
plt.ylabel('Blue Intensity')

plt.tight_layout()
plt.show()

# Advanced Visualization: Histogram Equalization
# Equalize the histogram for each channel
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
image_equalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# Display the original and equalized images side by side
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(image_equalized)
plt.title('Histogram Equalized Image')

plt.tight_layout()
plt.show()
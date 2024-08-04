import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate and plot histograms
def plot_histograms(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into RGB channels
    r, g, b = cv2.split(image_rgb)

    # Calculate histograms for each channel
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    # Plot the histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(hist_r, color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.plot(hist_g, color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.plot(hist_b, color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Replace 'your_image.jpg' with the path to your image file
plot_histograms('example.jpg')

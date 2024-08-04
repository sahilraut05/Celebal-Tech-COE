from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Set up credentials and endpoint
subscription_key = "YOUR_SUBSCRIPTION_KEY"
endpoint = "YOUR_COMPUTER_VISION_ENDPOINT"

# Initialize the Computer Vision client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Load and display the image to analyze
image_url = "URL_OF_IMAGE_TO_ANALYZE"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()

# Analyze the image
image_analysis = computervision_client.analyze_image(image_url, visual_features=[VisualFeatureTypes.tags])

# Display the tags detected
for tag in image_analysis.tags:
    print(f"{tag.name} ({tag.confidence * 100:.2f}%)")

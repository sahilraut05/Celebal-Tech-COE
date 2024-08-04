from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Set up credentials and endpoint
prediction_key = "YOUR_PREDICTION_KEY"  # Replace with your Custom Vision prediction key
endpoint = "YOUR_CUSTOM_VISION_ENDPOINT"  # Replace with your Custom Vision endpoint
project_id = "YOUR_PROJECT_ID"  # Replace with your project ID
published_name = "YOUR_PUBLISHED_NAME"  # Replace with your published iteration name

# Initialize the prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, credentials)

# Load and display the image to be predicted
image_url = "URL_OF_IMAGE_TO_PREDICT"  # Replace with the actual URL of the image
response = requests.get(image_url)

if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Predict the objects in the image
    results = predictor.classify_image_url(project_id, published_name, url=image_url)

    # Display the results
    for prediction in results.predictions:
        print(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

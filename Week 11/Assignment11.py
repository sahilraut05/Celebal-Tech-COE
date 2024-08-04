from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Set up credentials and endpoint
prediction_key = "69b4877580bf4f76b46b2b1554206358"
endpoint = "https://celebaltech.cognitiveservices.azure.com/"
project_id = "a850bab6-e090-499b-8103-e38fab2d0272"
published_name = "Simpson"

# Initialize the prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, credentials)

# Load and display the image to be predicted
image_url = "https://upload.wikimedia.org/wikipedia/en/a/aa/Bart_Simpson_200px.png"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()

try:
    # Predict the objects in the image
    results = predictor.classify_image_url(project_id, published_name, url=image_url)

    # Display the results
    for prediction in results.predictions:
        print(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
except Exception as e:
    print(f"An error occurred: {e}")

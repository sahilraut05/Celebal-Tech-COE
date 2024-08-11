from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Azure credentials
key = "48a220de52134d6c86596949287909d9"
endpoint = "https://assignment12.cognitiveservices.azure.com/"

# Function to authenticate the client
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return client

# Function to analyze sentiment
def sentiment_analysis_example(client):
    documents = ["The movie was fantastic! I enjoyed every minute of it."]
    response = client.analyze_sentiment(documents=documents)[0]
    print(f"Overall sentiment: {response.sentiment}")
    print(f"Positive: {response.confidence_scores.positive}")
    print(f"Neutral: {response.confidence_scores.neutral}")
    print(f"Negative: {response.confidence_scores.negative}")

# Main code
if __name__ == "__main__":
    client = authenticate_client()
    sentiment_analysis_example(client)

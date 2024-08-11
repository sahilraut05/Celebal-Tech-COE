import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Load Azure credentials from environment variables
key = os.getenv('AZURE_TEXT_ANALYTICS_KEY')
endpoint = os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')

def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return client

def sentiment_analysis_example(client):
    documents = ["The movie was fantastic! I enjoyed every minute of it."]
    response = client.analyze_sentiment(documents=documents)[0]
    print(f"Overall sentiment: {response.sentiment}")
    print(f"Positive: {response.confidence_scores.positive}")
    print(f"Neutral: {response.confidence_scores.neutral}")
    print(f"Negative: {response.confidence_scores.negative}")

if __name__ == "__main__":
    client = authenticate_client()
    sentiment_analysis_example(client)

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Explain the benefits of using Azure OpenAI Service"}
    ],
    max_tokens=150
)

print(response.choices[0].message['content'].strip())

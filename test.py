from dotenv import load_dotenv
load_dotenv()

import os
import openai

client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
response = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Hello"}]
)
print(response)

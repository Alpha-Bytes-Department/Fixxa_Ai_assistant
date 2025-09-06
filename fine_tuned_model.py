import openai
import speech_recognition as sr
import pyttsx3
import os

from dotenv import load_dotenv
from openai import OpenAI
# 1️⃣ OpenAI API key
# Load .env file
load_dotenv()
# Get the API key from environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# 2️⃣ System message
system_message = {
    "role": "system",
    "content": "You are Fixa, a friendly assistant helping service providers create quotes for their clients. "
}

# 3️⃣ Few-shot examples
few_shot_examples = [
    {"role": "user", "content": "Hello, I'm Fixa, your assistant. How can I help today?"},
    {"role": "assistant", "content": "I need a quote for plumbing."},
    {"role": "user", "content": "Sure! What's your client's name?"},
    {"role": "assistant", "content": "John Doe."},
]

# 4️⃣ Initialize text-to-speech engine
def ask_fixa(user_input):
    messages = [system_message] + few_shot_examples + [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_completion_tokens=500
    )
    return response.choices[0].message.content.strip()

print(ask_fixa("I need a quote for plumbing."))

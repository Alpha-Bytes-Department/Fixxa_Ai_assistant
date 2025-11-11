import os
import requests
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
import re
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional

# Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
client = OpenAI(api_key=api_key)
supabase = create_client(supabase_url, supabase_key)

# Transcription function
def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",  # Use standard Whisper model
            file=audio_file,
            language="en"
        )
    return transcript.text

# Client Data Schema
class ClientDetails(BaseModel):
    client_name: str
    contact_info: Optional[EmailStr] = None
    phone: Optional[str] = None
    service_type: str
    description: Optional[str] = None
    estimated_cost: Optional[float] = None

    @field_validator("estimated_cost", mode="before")
    def extract_number(cls, v):
        if isinstance(v, str):
            match = re.search(r"\d+(\.\d+)?", v)
            if match:
                return float(match.group())
        return v

    @field_validator("phone", mode="before")
    def extract_phone(cls, v):
        if isinstance(v, str):
            digits = re.sub(r"\D", "", v)
            if digits:
                return digits
        return v

# Extractor function
def extractor(text: str):
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "You are an expert at structured data extraction. Convert the transcription into structured JSON."
            },
            {"role": "user", "content": text}
        ],
        text_format=ClientDetails,
    )
    return response.output_parsed

def fetch_and_process_audio(url: str):
    print(f"Fetching audio from: {url}")

    # Download the audio
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download audio: {response.status_code}")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    try:
        # Transcribe
        transcription = transcribe_audio(temp_file_path)

        # Extract data
        client_data = extractor(transcription)

        print("Transcription:", transcription)
        print("Client Data:", client_data)

        return client_data

    finally:
        # Cleanup temp file
        os.remove(temp_file_path)

# URL for the first audio in quote_audio folder
url = "https://hnvtfxhapzjvglozozds.supabase.co/storage/v1/object/public/audio_storage/quote_audio/Recording.m4a"

if __name__ == "__main__":
    try:
        data = fetch_and_process_audio(url)
        print("Required Data:", data)
    except Exception as e:
        print("Error:", e)
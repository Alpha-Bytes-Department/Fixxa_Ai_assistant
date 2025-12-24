from fastapi import FastAPI, HTTPException

import openai
import sounddevice as sd
import wavio
import numpy as np
import time
from collections import deque

from openai import OpenAI
from pydantic import BaseModel

from openai import OpenAI
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import re

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Import Supabase audio functions
from supabase_audio import get_most_recent_audio, download_audio_file
import requests

#-------------------------------API key setup-------------------------------#
import os
from dotenv import load_dotenv
from openai import OpenAI
# 1️⃣ OpenAI API key
# Load .env file
load_dotenv()
# Get the API key from environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
#-------------------------------API key setup end-------------------------------#

app = FastAPI()



def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="en",
            prompt=""""My client's name is David and his email address is david.lee@email.com
 and his number is 555-4321-0987 and his address is 456 Oak Avenue, Riverside. The job I'll be doing with him is plumbing services, and the description of the job will be replacing pipes in the bathroom and kitchen. The estimated cost will be 1,200 dollars."

"My client's name is Sarah and her email address is sarah.williams@outlook.com
 and her number is 555-1234-5678 and her address is 789 Pine Road, Hilltop. The job I'll be doing with her is electrical wiring, and the description of the job will be installing new electrical wiring for the home’s new extension. The estimated cost will be 2,000 dollars."

"My client's name is Mark and his email address is mark.brown@gmail.com
 and his number is 987-654-3210 and his address is 101 Sunset Boulevard, Oceanview. The job I'll be doing with him is roof repair, and the description of the job will be fixing the leaky roof and replacing damaged shingles. The estimated cost will be 1,500 dollars."

"My client's name is Lucy and her email address is lucy.taylor@aol.com
 and her number is 444-777-8888 and her address is 202 Birch Drive, Forest Glen. The job I'll be doing with her is landscaping, and the description of the job will be designing and planting a new garden with flower beds and trees. The estimated cost will be 2,500 dollars."

"My client's name is Michael and his email address is michael.scott@dundermifflin.com
 and his number is 555-2356-7890 and his address is 100 Scranton Business Park, Scranton. The job I'll be doing with him is office renovation, and the description of the job will be renovating the office space with new furniture and lighting. The estimated cost will be 10,000 dollars."

"My client's name is Rachel and her email address is rachel.green@gmail.com
 and her number is 333-123-4567 and her address is 50 Central Perk, New York. The job I'll be doing with her is interior design, and the description of the job will be redesigning the living room with new decor, furniture, and a modern look. The estimated cost will be 4,000 dollars.
 
 "I’d like to see John Doe’s details."

"Could you pull up Sarah Williams' details for me?"

"Can you show me Mark Brown’s details from the roof repair job?"

"I need the details for Emma Johnson's home renovation job."

"I’d like to get Rachel Green's details to follow up on the living room redesign."
 

 
 """
        )

    return transcript.text
#-------------------------------Speech to text function end------------------------------#



#-------------------------------text extractor settings start-------------------------------#


class ClientDetails(BaseModel):
    client_name: str
    contact_info: Optional[EmailStr] = None
    phone: Optional[str] = None
    service_type: str
    description: Optional[str] = None
    estimated_cost: Optional[float] = None
    # send_quote_via: Optional[str] = None  # "gmail" or "whatsapp"

    # Extract numeric values for cost (e.g., "$12 per hour" -> 12.0)
    @field_validator("estimated_cost", mode="before")
    def extract_number(cls, v):
        if isinstance(v, str):
            match = re.search(r"\d+(\.\d+)?", v)
            if match:
                return float(match.group())
        return v

    # Extract digits for phone (e.g., "Call me at 0197-4273" -> "01974273")
    @field_validator("phone", mode="before")
    def extract_phone(cls, v):
        if isinstance(v, str):
            digits = re.sub(r"\D", "", v)
            if digits:
                return digits
        return v
# response = """My client's name is Sarah and her email address is sarah.williams@outlook.com
#     and her number is 555-1234-5678 and her address is 789 Pine Road, Hilltop. The job I'll be doing with her is electrical wiring, and the description of the job will be installing new electrical wiring for the home’s new extension. The estimated cost will be 2,000 dollars."""

#-------------------------------text extractor settings end-------------------------------#


#-------------------------------text extractor Function start-------------------------------#
def extractor(text: str):
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "You are an expert at structured data extraction. You will be given unstructured text from a user's speech and you should convert it into the given structure.",
            },
            {"role": "user", "content": """My client's name is Sarah and her email address is sarah.williams@outlook.com
    and her number is 555-1234-5678 and her address is 789 Pine Road, Hilltop. The job I'll be doing with her is electrical wiring, and the description of the job will be installing new electrical wiring for the home’s new extension. The estimated cost will be 2,000 dollars."""},
            {"role": "assistant", "content": """
    Structured Data: {
    "client_name": "Sarah Williams",
    "contact_info": "sarah.williams@outlook.com",
    "phone": "555-1234-5678",
    "service_type": "Electrical Wiring",
    "description": "Installing new electrical wiring for the home’s new extension",
    "estimated_cost": 2000
    }
    """
             },
             {"role": "user", "content": text},
        ],
        text_format=ClientDetails,
    )

    return response.output_parsed

#-------------------------------text extractor Function end-------------------------------#

#-------------------------------Api call function-------------------------------#

class StatusRequest(BaseModel):
    status: str  # "quote" or "invoice"


@app.post("/ProcessAudio")
async def ProcessAudio(request: StatusRequest):
    """
    Fetch audio from Supabase based on status, transcribe it, and extract client data.
    
    Args:
        request: StatusRequest with status field ("quote" or "invoice")
    
    Returns:
        Dictionary with transcription and extracted client data
    """
    try:
        # Step 1: Fetch the most recent audio file from Supabase
        audio_info = get_most_recent_audio(request.status)
        
        if not audio_info:
            raise HTTPException(status_code=404, detail=f"No audio file found for status: {request.status}")
        
        # Step 2: Download the audio file from the signed URL
        audio_url = audio_info['url']
        local_filename = f"temp_{audio_info['name']}"
        
        response = requests.get(audio_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download audio file from Supabase")
        
        # Save the audio file locally
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        
        # Step 3: Transcribe the audio
        transcription = transcribe_audio(local_filename)
        
        # Step 4: Extract client data from transcription
        client_data = extractor(transcription)
        
        # Clean up: delete temporary file
        if os.path.exists(local_filename):
            os.remove(local_filename)
        
        # Step 5: Return the results
        return {
            "status": request.status,
            "audio_file": audio_info['name'],
            "transcription": transcription,
            "client_data": client_data
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.get("/FixaAssist")
async def FixaAssist():
    # Step 1: Record Audio
    audio_file = record_audio()
    
    # Step 2: Transcribe Audio
    transcription = transcribe_audio(audio_file)
    client_data = extractor(transcription)
    
    # Step 3: Return the transcribed text
    return {"transcription": transcription, "client_data": client_data}

# if __name__ == "__main__":
#     audio_file = record_audio()
#     text = transcribe_audio(audio_file)
#     print("Transcription:", text)


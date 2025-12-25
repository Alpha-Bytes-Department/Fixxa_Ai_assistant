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
from typing import Optional, List
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

@app.get("/")
async def root():
    """
    Root endpoint - API information and available endpoints
    """
    return {
        "message": "Welcome to Fixxa AI Assistant API",
        "version": "1.0",
        "endpoints": {
            "POST /ProcessAudio": "Process audio from Supabase (requires status: 'quote' or 'invoice')",
            "GET /FixaAssist": "Record and transcribe audio locally",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "status": "online"
    }



def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="en",
            prompt=""""
            
            Hello there, i would like to make a quote on my client David Lee. He can be reached at david.lee@email.com
 or 555-4321-0987, and his address is 456 Oak Avenue, Riverside. The main service for this job is plumbing services, specifically replacing worn-out pipes in both the bathroom and kitchen, with an estimated total cost of $1,200. The service line-item focuses on pipe replacement under plumbing services at $10/hr for a duration of a few days, allowing enough time to remove old piping and fit the new lines properly. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20), depending on how many joints and turns the pipe route needs.

Hello there, i would like to make a invoice on my client Sarah Williams and she can be reached at sarah.williams@outlook.com
 or 555-1234-5678, and her address is 789 Pine Road, Hilltop. The main service for this job is electrical wiring, specifically installing fresh wiring for the home's new extension, with an estimated total cost of $2,000. The service line-item covers new wiring installation under electrical wiring at $10/hr for a duration of a few days, including safely routing cables, setting up connection points, and ensuring everything is ready for final testing. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20), which may be used for conduit routing and securing the setup.

Hello there, i would like to make a quote on my clientMark Brown. He will be available in mark.brown@gmail.com
 or 987-654-3210, and his address is 101 Sunset Boulevard, Oceanview. The main service for this job is roof repair, specifically fixing a leaky roof and replacing damaged shingles, with an estimated total cost of $1,500. The service line-item includes leak fixing and shingle replacement under roof repair at $10/hr for a duration of a few days, ensuring the leak source is sealed before new shingles are placed for a clean finish. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20), which can support temporary fastening, securing, or minor structural patchwork during the repair.

Hello there, i would like to make a quote on my client Lucy Taylor. She can be reached at lucy.taylor@aol.com
 or 444-777-8888, and her address is 202 Birch Drive, Forest Glen. The main service for this job is landscaping, specifically designing and planting a new garden with flower beds and trees, with an estimated total cost of $2,500. The service line-item includes garden design and planting under landscaping at $10/hr for a duration of a few days, covering layout planning, soil preparation, planting, and tidy finishing touches. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20), which can be used for basic garden support, light irrigation routing, or securing elements in place.

Hello there, i would like to make a quote on my client Michael Scott. He can be reached at michael.scott@dundermifflin.com
 or 555-2356-7890, and his address is 100 Scranton Business Park, Scranton. The main service for this job is office renovation, specifically upgrading the office space with new furniture and improved lighting, with an estimated total cost of $10,000. The service line-item includes office renovation work under office renovation at $10/hr for a duration of a few days, allowing time for rearranging the workspace, installing lighting fixtures, and completing a clean setup for employees. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20), which may support wiring organization, fixture mounting, or minor installation adjustments.

Hi, i would like to make a quote on my client Rachel Green. She can be reached at rachel.green@gmail.com
 or 333-123-4567, and her address is 50 Central Perk, New York. The main service for this job is interior design, specifically redesigning the living room with fresh decor, new furniture, and a modern look, with an estimated total cost of $4,000. The service line-item includes living room redesign under interior design at $10/hr for a duration of a few days, covering planning, layout refinement, styling decisions, and final placement for a cohesive modern finish. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20), which can help with light fixture setup, mounting, or small installation needs as part of the refresh.

 
 """
        )

    return transcript.text
#-------------------------------Speech to text function end------------------------------#



#-------------------------------text extractor settings start-------------------------------#


class ServiceItem(BaseModel):
    description: str
    service: str
    rate: str
    duration: str


class MaterialItem(BaseModel):
    material: str
    quantity: str | int  # Allow both string (e.g., "2-4") and int
    unit_price: str
    amount: str


class ClientDetails(BaseModel):
    status: str
    client_name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    service_type: str
    job_description: Optional[str] = None
    estimated_cost: Optional[float] = None
    estimated_cost_currency: Optional[str] = None  # Added currency field
    services: Optional[List[ServiceItem]] = None
    materials: Optional[List[MaterialItem]] = None

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

#-------------------------------text extractor settings end-------------------------------#


#-------------------------------text extractor Function start-------------------------------#
def extractor(text: str):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at structured data extraction. You will be given unstructured text from a user's speech and you should convert it into the given structure. Extract ALL information including status (quote/invoice), client details, service information, service line-items, materials, and currency.",
            },
            {"role": "user", "content": """I would like to make an invoice on my client David Lee. He can be reached at david.lee@email.com or 555-4321-0987, and his address is 456 Oak Avenue, Riverside. The main service for this job is plumbing services, specifically replacing pipes in the bathroom and kitchen, with an estimated total cost of $1,200. The service line-item focuses on pipe replacement under plumbing services at $10/hr for a duration of a few days. The materials required are cable (quantity 1, unit price $5, estimated amount $5), pipes (quantity 2, unit price $5, estimated amount $10), and fittings (quantity 2–4, unit price $5, estimated amount $10–$20)."""},
            {"role": "assistant", "content": """{"status": "invoice", "client_name": "David Lee", "email": "david.lee@email.com", "phone": "5554321098", "address": "456 Oak Avenue, Riverside", "service_type": "Plumbing Services", "job_description": "Replacing pipes in the bathroom and kitchen", "estimated_cost": 1200, "estimated_cost_currency": "USD", "services": [{"description": "Replace pipes in bathroom and kitchen", "service": "plumbing services", "rate": "10/hr", "duration": "few days"}], "materials": [{"material": "Cable", "quantity": 1, "unit_price": "$5", "amount": "$5"}, {"material": "Pipes", "quantity": 2, "unit_price": "$5", "amount": "$10"}, {"material": "Fittings", "quantity": "2-4", "unit_price": "$5", "amount": "$10-$20"}]}"""},
            {"role": "user", "content": text},
        ],
        response_format=ClientDetails,
    )

    return completion.choices[0].message.parsed

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

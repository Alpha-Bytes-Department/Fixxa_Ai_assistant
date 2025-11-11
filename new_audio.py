from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os, re
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from openai import OpenAI

# ------------------ API Key Setup ------------------ #
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI()

# ------------------ Transcription ------------------ #
def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="en"
        )
    return transcript.text

# ------------------ Client Data Schema ------------------ #
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

# ------------------ Extractor ------------------ #
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

def FixaAssist(audio: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_location = f"temp_{audio.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Transcribe
        transcription = transcribe_audio(file_location)

        # Extract client details
        client_data = extractor(transcription)

        # Cleanup
        os.remove(file_location)

        return JSONResponse(content={
            "transcription": transcription,
            "client_data": client_data
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# ------------------ API Endpoint ------------------ #
# @app.post("/FixaAssist")
# async def FixaAssist(audio: UploadFile = File(...)):
#     try:
#         # Save uploaded file temporarily
#         file_location = f"temp_{audio.filename}"
#         with open(file_location, "wb") as buffer:
#             shutil.copyfileobj(audio.file, buffer)

#         # Transcribe
#         transcription = transcribe_audio(file_location)

#         # Extract client details
#         client_data = extractor(transcription)

#         # Cleanup
#         os.remove(file_location)

#         return JSONResponse(content={
#             "transcription": transcription,
#             "client_data": client_data
#         })

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
if __name__ == "__main__":
    
    data=FixaAssist('C:\fixa_assist\recorded_audio.wav')
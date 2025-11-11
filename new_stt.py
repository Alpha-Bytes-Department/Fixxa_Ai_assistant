from fastapi import FastAPI

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

#-------------------------------audio recording settings-------------------------------#
fs = 16000
channels = 1

# VAD knobs
_CHUNK = 1024                 # ~64 ms at 16 kHz
_THRESHOLD = 0.008       # lower => more sensitive. Try 0.006–0.012 as needed
_SILENCE_LIMIT_SEC = 0.5      # stop after sustained silence
_START_CONFIRM_FRAMES = 4     # need ~5 chunks (~320 ms) above threshold to start
_PREROLL_SEC = 0.5            # keep ~0.5s before start so first word isn’t clipped
_SMOOTH_N = 3                 # rolling avg over last N energies to reduce jitter

#-------------------------------audio recording settings-------------------------------#


#-------------------------------Speech to text function start------------------------------#
def _energy(x: np.ndarray) -> float:
    # mean absolute amplitude (RMS would work too)
    return float(np.mean(np.abs(x)))

def record_audio():
    print("Speak to start. I’ll stop automatically after silence.")
    recording = []
    started = False
    silence_start = None
    start_counter = 0

    preroll_frames = int(_PREROLL_SEC * fs // _CHUNK) or 1
    preroll = deque(maxlen=preroll_frames)

    energy_hist = deque(maxlen=_SMOOTH_N)

    while True:
        chunk = sd.rec(_CHUNK, samplerate=fs, channels=channels, dtype="float32")
        sd.wait()
        chunk = chunk.flatten()

        # smooth energy
        energy_hist.append(_energy(chunk))
        energy = sum(energy_hist) / len(energy_hist)

        if not started:
            preroll.append(chunk)
            if energy > _THRESHOLD:
                start_counter += 1
            else:
                start_counter = 0

            if start_counter >= _START_CONFIRM_FRAMES:
                started = True
                print("Voice detected — recording…")
                # dump preroll into output
                recording.extend(list(preroll))
                start_counter = 0
                silence_start = None
        else:
            recording.append(chunk)
            if energy <= _THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= _SILENCE_LIMIT_SEC:
                    print("Silence detected — stopping.")
                    break
            else:
                silence_start = None

    audio = np.concatenate(recording, axis=0) if recording else np.zeros((0,), dtype="float32")
    filename = "recorded_audio.wav"
    wavio.write(filename, audio, fs, sampwidth=2)  # 16-bit PCM
    return filename

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


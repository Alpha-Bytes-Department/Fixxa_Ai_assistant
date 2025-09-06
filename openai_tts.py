import openai
import tempfile
from pydub import AudioSegment
from pydub.playback import play

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


def speak(text):
    # 2️⃣ Create TTS audio (returns mp3)
    audio_response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="sage",
        input=text
    )

    # 3️⃣ Save the audio to a temporary mp3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_response.read())
        mp3_file_path = tmp_file.name

    # 4️⃣ Convert mp3 to WAV for playback
    sound = AudioSegment.from_file(mp3_file_path, format="mp3")
    play(sound)


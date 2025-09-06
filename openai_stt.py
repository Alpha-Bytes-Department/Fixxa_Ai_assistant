
import openai
import sounddevice as sd
import wavio

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




# 2️⃣ Recording settings
fs = 16000  # Sampling rate
channels = 1

def record_audio():
    print("Press Enter to start recording, and Ctrl+C or stop speaking to end.")
    input("Press Enter to start...")
    print("Recording... Speak now!")

    recording = []
    try:
        while True:
            # Record in chunks of 1 second
            chunk = sd.rec(int(fs * 1), samplerate=fs, channels=channels)
            sd.wait()
            recording.append(chunk)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    # Combine chunks
    audio_data = np.vstack(recording)
    filename = "recorded_audio.wav"
    wavio.write(filename, audio_data, fs, sampwidth=2)
    return filename

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text





if __name__ == "__main__":
    import numpy as np
    audio_file = record_audio()
    text = transcribe_audio(audio_file)
    print("Transcription:", text)

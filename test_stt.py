import openai
import sounddevice as sd
import wavio
import numpy as np
import time
from collections import deque

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


fs = 16000
channels = 1

# VAD knobs
_CHUNK = 1024                 # ~64 ms at 16 kHz
_THRESHOLD = 0.008       # lower => more sensitive. Try 0.006–0.012 as needed
_SILENCE_LIMIT_SEC = 0.5      # stop after sustained silence
_START_CONFIRM_FRAMES = 4     # need ~5 chunks (~320 ms) above threshold to start
_PREROLL_SEC = 0.5            # keep ~0.5s before start so first word isn’t clipped
_SMOOTH_N = 3                 # rolling avg over last N energies to reduce jitter

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
            model="whisper-1",
            file=audio_file,
            language="en",          # bias to English
            # prompt="This is an invoice workflow assistant.",  # optional domain bias
        )
    return transcript.text

if __name__ == "__main__":
    audio_file = record_audio()
    text = transcribe_audio(audio_file)
    print("Transcription:", text)

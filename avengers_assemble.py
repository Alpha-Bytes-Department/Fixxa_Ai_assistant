import fine_tuned_model
import openai_stt
import openai_tts 
import test_stt
import test_model
import perfect_ai



def tts_loop():
    audio_file = test_stt.record_audio()
    text = test_stt.transcribe_audio(audio_file)
    return text

if __name__ == "__main__":
    
    while True:
        
        user_input = tts_loop()
        print(f"You: {user_input}")
        reply = perfect_ai.fixa_ai(user_input)
        structured_data = reply[1]
        print("Structured Data:", structured_data.model_dump_json(indent=2))
        print(f"Fixa: {reply[0]}")
        openai_tts.speak(reply[0])



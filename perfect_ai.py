from openai import OpenAI
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import re
import test_stt
import openai_tts

# -------------------------------
# Initialize OpenAI client
# -------------------------------
client = OpenAI()
MODEL = "ft:gpt-4.1-mini-2025-04-14:fixxa-ai::C8hQF4PN"

# -------------------------------
# Pydantic model for structured data
# -------------------------------
class ClientDetails(BaseModel):
    client_name: str
    contact_info: Optional[EmailStr] = None
    phone: Optional[str] = None
    service_type: str
    description: Optional[str] = None
    estimated_cost: Optional[float] = None
    send_quote_via: Optional[str] = None  # "gmail" or "whatsapp"

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

# -------------------------------
# Extractor function
# -------------------------------
def extractor(text: str) -> ClientDetails:
    """
    Extract structured client/service info from plain text input.
    Returns a ClientDetails object.
    """
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": """You are a data extractor. Extract all client/service details into these fields:
Structured Data: {
  "client_name": "",
  "contact_info": null,
  "phone": null,
  "service_type": "",
  "description": null,
  "estimated_cost": null,
  "send_quote_via": ""
}
- Only return valid JSON. Fill only available fields. Also check if the user has provided any other fields in the same message then dont ask for that field again"""
            },
            {"role": "user", "content": text},
        ],
        text_format=ClientDetails,
        temperature=0,
    )
    return response.output_parsed

# -------------------------------
# Merge new extracted data into master state
# -------------------------------
def merge_data(master: ClientDetails, new_data: ClientDetails) -> ClientDetails:
    for field in master.model_fields:
        value = getattr(new_data, field)
        if value not in (None, "", []):
            setattr(master, field, value)
    return master

# -------------------------------
# Chat history and master state
# -------------------------------

chat_history = [
    {"role": "system", "content": "You are Fixa, a friendly assistant helping service providers or the users create quotes for their clients. You would gather details like the user's client name, contact information, service type, description and estimated cost. Do not ask for any other fields. Keep the conversation friendly, jolly, and concise. Ensure the user knows what information needed and you are asking for it. Start the conversation by greeting the user and asking how you can help them today. User information is provided here in the triple backticks. Instead of calling the user [Service Provider]. Use the information in the triple backticks to call the user by name.```[User name : Shoshi]```. Also after the user says what they want and you say lets get started, start right away after that. Like, User : I would like to make an invoice. You : Sure! Lets get started. What's the client's full name?, User : I would like to search for a previous customer. You : Sure! I can fetch the data. What's the customer's name?. IF the user says nothing, Find out what the user want. If the user says they want to make a quote, start asking for the fields one by one. Also check if the user has provided any other fields in the same message then dont ask for that field again. Question repetion is must be avoided unless you don't understand the user."},
    {"role":"user","content":"I'd like to make a quote"},
    {"role":"assistant","content":"Great! Let's get started. What’s your client’s full name?"},
    {"role":"user","content":""},
    {"role":"assistant","content":"How can I assist you today? Do you need help creating a quote for a service? Or view a previous customer?"}
]

master_data = ClientDetails(
    client_name="",
    contact_info=None,
    phone=None,
    service_type="",
    description=None,
    estimated_cost=None,
    send_quote_via="",
)

# -------------------------------
# Main function
# ------------------------------


def fixa_ai(user_input: str):
    global master_data

    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Generate AI response for conversation
    response = client.chat.completions.create(
        model=MODEL,
        messages=chat_history,
        temperature=0.3,
        max_tokens=500,
    )
    reply = response.choices[0].message.content

    # Append AI reply to chat history
    chat_history.append({"role": "assistant", "content": reply})


    # Extract structured info from **user input**
    new_data = extractor(user_input)

    # Merge into master_data
    master_data = merge_data(master_data, new_data)

    # Print updated structured data
    # print("Structured Data:", master_data.model_dump_json(indent=2))
    return reply,master_data

def tts_loop():
    audio_file = test_stt.record_audio()
    text = test_stt.transcribe_audio(audio_file)
    return text

# -------------------------------
# Run multi-turn conversation
# -------------------------------
while True:

    user_input = tts_loop()
    print(f"You: {user_input}")
    if "bye" in user_input.lower() :
        reply = fixa_ai(user_input)
        print(f"Fixa: {reply[0]}")
        break
    reply = fixa_ai(user_input)
    print(f"Fixa: {reply[0]}")
    openai_tts.speak(reply[0])
    # print("Structured Data:", reply[1].model_dump_json(indent=2))
    # openai_tts.speak(reply[0])

print("\nFinal Structured Data:", master_data.model_dump_json(indent=2))

# if __name__ == "__main__":
    
#     while True:
        
#         user_input = tts_loop()
#         print(f"You: {user_input}")
#         reply = fixa_ai(user_input)
#         structured_data = reply[1]
#         print("Structured Data:", structured_data.model_dump_json(indent=2))
#         openai_tts.speak(reply[0])


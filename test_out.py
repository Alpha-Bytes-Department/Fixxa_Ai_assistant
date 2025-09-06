from openai import OpenAI
from pydantic import BaseModel

from openai import OpenAI
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import re

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

client = OpenAI()
app = FastAPI()

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
response = """My client's name is Sarah and her email address is sarah.williams@outlook.com
    and her number is 555-1234-5678 and her address is 789 Pine Road, Hilltop. The job I'll be doing with her is electrical wiring, and the description of the job will be installing new electrical wiring for the home’s new extension. The estimated cost will be 2,000 dollars."""
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


@app.get("/client_data", response_model=ClientDetails)
async def get_client_data():
    # Example instance
    client_data = extractor(response)
    
    return client_data

# print(extractor(response))
# print(type(extractor(response)))
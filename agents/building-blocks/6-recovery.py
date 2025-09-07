"""
Recovery: Manages failures and exceptions gracefully in agent workflows.
This component implements retry logic, fallback processes, and error handling to ensure system resilience.
"""

import os
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env file

class UserInfo(BaseModel):
    name: str
    email: str
    age: Optional[int] = None  # Optional field


def resilient_intelligence(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get structured output
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "Extract user information from the text."},
            {"role": "user", "content": prompt},
        ],
        text_format=UserInfo,
        temperature=0.0,
    )

    user_data = response.output_parsed.model_dump()

    try:
        # Try to access age field and check if it's valid
        age = user_data["age"]
        if age is None:
            raise ValueError("Age is None")
        
        name = user_data["name"]
        if name is None or name.strip() == "":
            raise ValueError("Name is None")
        
        email = user_data["email"]
        if email is None:
            raise ValueError("Email is None")

        return name, age, email

    except (KeyError, TypeError, ValueError):
        if ValueError == "Age is None":
            print("❌ Age not available, using fallback info...")
            return f"User {user_data['name']} has email {user_data['email']}"
        elif ValueError == "Name is None":
            print("❌ Name not available, using fallback info...")
            return f"User with email {user_data['email']} is {user_data['age']} years old"
        elif ValueError == "Email is None":
            print("❌ Email not available, using fallback info...")
            return f"User {user_data['name']} is {user_data['age']} years old"
        else:
            print("❌ Critical info missing, cannot proceed.")
            return "Insufficient user information provided."


if __name__ == "__main__":
    result = resilient_intelligence(
        "I am 38 and my email is john@example.com"
    )
    print("Recovery Output:")
    print(result)

"""
Memory: Stores and retrieves relevant information across interactions.
This component maintains conversation history and context to enable coherent multi-turn interactions.

More info: https://platform.openai.com/docs/guides/conversation-state?api-mode=responses
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_joke(prompt: str):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text, response.id


def ask_followup(prompt: str, previous_response_id: str):
    response = client.responses.create(
        model="gpt-4o-mini",
        previous_response_id=previous_response_id,
        input=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text, response.id


def ask_another_followup(prompt: str, previous_response_id: str):
    
    response = client.responses.create(
        model="gpt-4o-mini",
        previous_response_id=previous_response_id,
        input=[
            {"role": "user", "content": prompt},
        ],
    )

    return response.output_text, response.id


if __name__ == "__main__":
    # First: Ask for a joke
    prompt = "Tell me a joke about programming"
    joke_response, response_id = ask_joke(prompt)
    print(joke_response, "\n")

    # Follow-up question    
    prompt = "Explain why this funny?"
    second_response, second_response_id = ask_followup(prompt, response_id)
    print(second_response)

    # Follow-up question    
    prompt = "What was my previous question?"
    third_response, third_response_id = ask_another_followup(prompt, second_response_id)
    print(third_response)

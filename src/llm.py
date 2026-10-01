import os

from dotenv import load_dotenv
from google import genai

from src.config import LLM_MODEL


load_dotenv()


def create_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY not found."
        )

    return genai.Client(
        api_key=api_key
    )


def generate_answer(
    client,
    query,
    context
):

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the provided context.

Rules:
- Do not use outside knowledge.
- If the answer cannot be found in the context, say:
  "I could not find the answer in the provided document."
- Keep the answer clear and concise.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.interactions.create(
        model=LLM_MODEL,
        input=prompt
    )

    return response.output_text
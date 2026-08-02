"""Groq LLM wrapper."""

import os

from groq import Groq

from dotenv import load_dotenv


load_dotenv()
MODEL_NAME = "openai/gpt-oss-120b"


class LLM:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not found."
            )

        self.client = Groq(api_key=api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate an answer from the LLM.
        """

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content.strip()
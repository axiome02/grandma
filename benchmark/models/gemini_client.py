import os
from typing import Optional

import google.generativeai as genai
from PIL import Image

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models (Gemini 1.5, 2.0, 2.5, etc.)."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name)
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_name)

    def query(self, question: str, image_path: Optional[str] = None) -> dict:
        """Returns {"text": str, "input_tokens": int, "output_tokens": int}"""
        parts = []

        if image_path:
            image = Image.open(image_path)
            parts.append(image)

        parts.append(question)

        response = self.model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(temperature=0),
        )

        usage = response.usage_metadata
        return {
            "text": response.text.strip(),
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
        }

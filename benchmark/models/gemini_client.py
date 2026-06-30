import base64
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

    def query(self, question: str, image_path: Optional[str] = None) -> str:
        parts = []

        if image_path:
            image = Image.open(image_path)
            parts.append(image)

        parts.append(question)

        response = self.model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(temperature=0),
        )
        return response.text.strip()

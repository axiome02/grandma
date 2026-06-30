import base64
import os
from typing import Optional

from openai import OpenAI

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def query(self, question: str, image_path: Optional[str] = None) -> str:
        messages = []

        if image_path:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_data}"},
                    },
                    {"type": "text", "text": question},
                ],
            })
        else:
            messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

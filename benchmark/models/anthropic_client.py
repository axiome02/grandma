import base64
import os
from typing import Optional

import anthropic

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic models (Claude 3.5, Claude 4, etc.)."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def query(self, question: str, image_path: Optional[str] = None) -> dict:
        """Returns {"text": str, "input_tokens": int, "output_tokens": int}"""
        content = []

        if image_path:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": image_data},
            })

        content.append({"type": "text", "text": question})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return {
            "text": response.content[0].text.strip(),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

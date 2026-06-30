from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients.

    All clients must support both text-only and multimodal (text + image) inputs,
    and must return token usage alongside the text response.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def query(self, question: str, image_path: Optional[str] = None) -> dict:
        """Send a question to the LLM and return response + token usage.

        Args:
            question: The question text to ask.
            image_path: Optional path to an image file for multimodal questions.

        Returns:
            A dict with keys:
                - "text" (str): The model's response.
                - "input_tokens" (int): Number of input/prompt tokens used.
                - "output_tokens" (int): Number of output/completion tokens used.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"

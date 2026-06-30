"""
__init__.py for benchmark.models

Auto-registers available model clients.
"""

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient

__all__ = ["BaseLLMClient", "OpenAIClient", "AnthropicClient", "GeminiClient"]

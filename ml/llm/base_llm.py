"""
Abstract base class for all LLM backends.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def load_model(self):
        """Load model into memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Generate text from prompt."""
        pass
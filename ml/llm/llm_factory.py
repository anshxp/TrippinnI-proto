"""
Factory for creating LLM backends.
"""

from ml.llm.huggingface_llm import HuggingFaceLLM


def create_llm():

    return HuggingFaceLLM()
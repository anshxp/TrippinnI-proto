"""
Hugging Face implementation of BaseLLM.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import torch

from config import (
    LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)

from ml.llm.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):

    def __init__(self):

        self.tokenizer = None
        self.model = None

        self.load_model()

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
"""
Utility functions for LLMs.
"""

import re


def clean_response(
    text: str,
) -> str:

    return text.strip()


def extract_json(
    text: str,
) -> str:

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if match:
        return match.group(0)

    return text
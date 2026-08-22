#################################################
# LLM Configuration
#################################################

LLM_MODEL = "Qwen/Qwen3-4B-Instruct"

MAX_NEW_TOKENS = 512

TEMPERATURE = 0.2

"""
Rule Validator.

Applies deterministic healthcare validation rules.
"""

from __future__ import annotations

from typing import Any

from models.detector_result import DetectorResult


class RuleValidator:
    """
    Executes rule-based validation.

    This class is used by the OutlierDetector.
    """

    def __init__(self) -> None:
        pass

    def validate(
        self,
        dataset: Any,
        profile: Any,
    ) -> DetectorResult:
        """
        Apply all configured validation rules.

        Returns
        -------
        DetectorResult
        """

        result = DetectorResult("RuleValidator")

        # TODO:
        # - Missing mandatory fields
        # - Date consistency
        # - Age validation
        # - Clinical ranges
        # - Referential integrity
        # - ICD consistency
        # - Medication consistency
        # - Unit validation

        return result
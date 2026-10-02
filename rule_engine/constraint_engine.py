"""Unified facade for modular ML constraint discovery."""
from __future__ import annotations
from typing import Any
import pandas as pd
from rule_engine.constraint_inference import ConstraintInferer

class ConstraintEngine:
    def __init__(self) -> None:
        self.inferer = ConstraintInferer()

    def infer(self, dataframe: pd.DataFrame, profile: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        return self.inferer.infer(dataframe, profile or {})

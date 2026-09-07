"""Base interface for quality detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models.detector_result import DetectorResult


class BaseDetector(ABC):
    """Common interface implemented by all quality detectors."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        """Detect data-quality issues and return a detector result."""
        raise NotImplementedError

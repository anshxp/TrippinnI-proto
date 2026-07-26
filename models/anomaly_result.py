from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AnomalyResult:
    row_index: int

    anomaly_score: float

    confidence: float

    detectors: List[str] = field(default_factory=list)

    scores: Dict[str, float] = field(default_factory=dict)

    severity: str = "MEDIUM"
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DuplicatePair:
    left_index: int
    right_index: int

    confidence: float

    methods: List[str] = field(default_factory=list)

    scores: Dict[str, float] = field(default_factory=dict)
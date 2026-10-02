"""Serializable provenance record for a learned constraint."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from typing import Any

@dataclass
class ConstraintRecord:
    constraint_type: str
    scope: str
    variables: list[str]
    condition: dict[str,Any]
    model_evidence: list[dict[str,Any]]=field(default_factory=list)
    support: float=0.0
    confidence: float=0.0
    population: dict[str,Any]=field(default_factory=dict)
    provenance: dict[str,Any]=field(default_factory=dict)
    validation_status: str="candidate"

    def to_dict(self)->dict[str,Any]:
        return asdict(self)

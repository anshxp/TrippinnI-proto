"""Healthcare terminology abstraction layer.

The prototype does not bundle or assume a specific terminology authority.
Adapters can later resolve concepts against SNOMED CT, LOINC, RxNorm, UCUM,
OMOP, FHIR terminology services, or organization-specific vocabularies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(slots=True)
class TerminologyMatch:
    source_value: str
    system: str
    code: str | None
    display: str | None
    confidence: float
    evidence: Dict[str, Any]


class TerminologyResolver(Protocol):
    """Interface implemented by a healthcare terminology adapter."""

    def resolve(self, value: str, *, context: Dict[str, Any] | None = None) -> TerminologyMatch | None:
        ...


class NoOpTerminologyResolver:
    """Safe default that never invents a terminology mapping."""

    def resolve(self, value: str, *, context: Dict[str, Any] | None = None) -> TerminologyMatch | None:
        return None


class TerminologyContext:
    """Attach optional terminology evidence to semantic context."""

    def __init__(self, resolver: TerminologyResolver | None = None) -> None:
        self.resolver = resolver or NoOpTerminologyResolver()

    def enrich(self, value: str, *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        match = self.resolver.resolve(value, context=context)
        if match is None:
            return {
                "resolved": False,
                "source_value": value,
                "reason": "no terminology adapter configured",
            }
        return {
            "resolved": True,
            "source_value": match.source_value,
            "system": match.system,
            "code": match.code,
            "display": match.display,
            "confidence": match.confidence,
            "evidence": match.evidence,
        }

"""Dataset-agnostic healthcare semantic context inference.

This layer deliberately infers *candidate* healthcare meaning from observed
schema/profile metadata. It does not assume MIMIC, OMOP, FHIR, or a specific
vendor schema, and it does not claim that a column is clinically authoritative.
"""

from __future__ import annotations

import re
from typing import Any, Dict


_IDENTIFIER_TOKENS = {
    "id", "identifier", "mrn", "medical_record", "patient_id", "person_id",
    "subject_id", "encounter_id", "visit_id", "admission_id", "stay_id",
    "provider_id", "organization_id", "facility_id", "payer_id", "device_id",
    "code", "concept_id",
}
_TEMPORAL_TOKENS = {
    "date", "time", "timestamp", "datetime", "start", "end", "admit",
    "admission", "discharge", "birth", "death", "onset", "stop", "recorded",
}
_CLINICAL_TOKENS = {
    "diagnosis", "procedure", "medication", "drug", "lab", "laboratory",
    "observation", "result", "vital", "symptom", "condition", "allergy",
    "immunization", "specimen", "encounter", "visit", "dose", "unit",
}


def _tokens(name: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", str(name)).lower()
    return {part for part in re.split(r"[^a-z0-9]+", normalized) if part}


def infer_column_context(column: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Infer conservative semantic context for one column."""
    tokens = _tokens(column)
    semantic_type = str(metadata.get("semantic_type", "unknown")).lower()
    hints = set()

    if semantic_type == "identifier" or tokens & _IDENTIFIER_TOKENS or column.lower().endswith("_id"):
        hints.add("identifier")
    if semantic_type == "datetime" or tokens & _TEMPORAL_TOKENS:
        hints.add("temporal")
    if tokens & _CLINICAL_TOKENS or semantic_type == "medical_code":
        hints.add("clinical")
    if "unit" in tokens or tokens & {"uom", "units"}:
        hints.add("unit")

    role = "clinical_data" if "clinical" in hints else "administrative_data"
    if "identifier" in hints:
        role = "entity_identifier"
    elif "temporal" in hints:
        role = "event_time"
    elif "unit" in hints:
        role = "measurement_unit"

    return {
        "column": column,
        "semantic_type": metadata.get("semantic_type", "unknown"),
        "roles": sorted(hints),
        "role": role,
        "inference": "schema/profile heuristic",
        "confidence": _confidence(hints, semantic_type),
    }


def _confidence(hints: set[str], semantic_type: str) -> float:
    if semantic_type in {"identifier", "datetime", "medical_code"}:
        return 0.90
    if hints:
        return 0.70
    return 0.40


class SemanticContextBuilder:
    """Build dataset-agnostic semantic context from profiling reports."""

    VERSION = "0.1"

    def build(self, profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        tables: Dict[str, Any] = {}
        for table, profile in profiles.items():
            columns = profile.get("columns", {}) if isinstance(profile, dict) else {}
            tables[table] = {
                "table": table,
                "columns": {
                    name: infer_column_context(name, metadata or {})
                    for name, metadata in columns.items()
                },
                "candidate_keys": profile.get("keys", {}) if isinstance(profile, dict) else {},
                "dataset": profile.get("dataset", {}) if isinstance(profile, dict) else {},
            }

        return {
            "context_version": self.VERSION,
            "scope": "dataset_agnostic_healthcare",
            "tables": tables,
        }

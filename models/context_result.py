"""Serializable output model for healthcare semantic context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class ContextResult:
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_context": self.semantic_context,
            "knowledge_graph": self.knowledge_graph,
            "summary": self.summary,
        }

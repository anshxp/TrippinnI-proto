"""Shared action specifications for deterministic preprocessing.

No model client, network request, Ollama server, or Qwen dependency is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    name: str
    module: str
    description: str
    risk: str = "low"
    requires_approval: bool = False


@dataclass
class ActionResult:
    action_id: str
    action: str
    module: str
    status: str
    summary: str
    before_profile: dict[str, Any]
    after_profile: dict[str, Any] | None = None
    changed_records: int = 0
    archived_records: int = 0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

"""Immutable-style audit records for preprocessing transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class TransformationRecord:
    table: str
    row_index: int | None
    column: str | None
    action: str
    reason: str
    original_value: Any = None
    new_value: Any = None
    method: str | None = None
    detector: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransformationLog:
    records: list[TransformationRecord] = field(default_factory=list)

    def add(self, record: TransformationRecord) -> None:
        self.records.append(record)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_count": self.count,
            "records": [record.to_dict() for record in self.records],
        }

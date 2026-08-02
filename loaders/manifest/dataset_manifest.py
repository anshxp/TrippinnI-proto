from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetManifest:
    """
    Metadata describing a dataset.

    This class contains descriptive information only.
    It does not perform any loading, validation, or parsing.
    """

    # Basic information
    dataset_name: str
    dataset_type: str
    dataset_version: str = "1.0"

    # Location
    root_path: Path | str = Path()

    # Supported file formats
    supported_formats: list[str] = field(default_factory=list)

    # Dataset metadata
    description: str = ""
    author: str = ""
    source: str = ""

    # Optional custom metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize path and formats."""

        self.root_path = Path(self.root_path)

        self.supported_formats = [
            fmt.lower().lstrip(".")
            for fmt in self.supported_formats
        ]

    @property
    def exists(self) -> bool:
        """Whether the dataset root exists."""
        return self.root_path.exists()

    @property
    def is_directory(self) -> bool:
        """Whether the dataset root is a directory."""
        return self.root_path.is_dir()

    def supports(self, extension: str) -> bool:
        """
        Check whether a file extension is supported.

        Example:
            manifest.supports("csv")
            manifest.supports(".json")
        """

        extension = extension.lower().lstrip(".")
        return extension in self.supported_formats

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a serializable dictionary."""

        return {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "dataset_version": self.dataset_version,
            "root_path": str(self.root_path),
            "supported_formats": self.supported_formats,
            "description": self.description,
            "author": self.author,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        """Create a manifest from a dictionary."""

        return cls(
            dataset_name=data["dataset_name"],
            dataset_type=data["dataset_type"],
            dataset_version=data.get("dataset_version", "1.0"),
            root_path=data["root_path"],
            supported_formats=data.get("supported_formats", []),
            description=data.get("description", ""),
            author=data.get("author", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )
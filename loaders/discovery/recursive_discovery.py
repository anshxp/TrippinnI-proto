from __future__ import annotations

from pathlib import Path


class RecursiveDiscovery:
    """
    Discovers supported files recursively.

    Example:

    dataset/
        hosp/
            patients.csv
            admissions.csv

        icu/
            chartevents.csv.gz

        fhir/
            bundle.json

    All supported files will be discovered.
    """

    DEFAULT_PATTERNS = (
        "*.csv",
        "*.csv.gz",
        "*.json",
    )

    def __init__(self, patterns: tuple[str, ...] | None = None):
        self.patterns = patterns or self.DEFAULT_PATTERNS

    def discover(self, directory: str | Path) -> list[Path]:
        """
        Discover supported files recursively.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(directory)

        files: list[Path] = []

        for pattern in self.patterns:
            files.extend(directory.rglob(pattern))

        return sorted(files)

    def supports(self, file_path: str | Path) -> bool:
        """
        Check whether the file extension is supported.
        """

        path = Path(file_path)

        suffixes = "".join(path.suffixes[-2:]).lower()

        if suffixes == ".csv.gz":
            return True

        return path.suffix.lower() in {
            ".csv",
            ".json",
        }
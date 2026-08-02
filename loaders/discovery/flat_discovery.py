from __future__ import annotations

from pathlib import Path


class FlatDiscovery:
    """
    Discovers supported files in a single directory.

    Non-recursive.

    Example:

    dataset/
        patients.csv
        encounters.csv
        observations.json

    It will NOT enter subdirectories.
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
        Discover all supported files in the given directory.
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
            files.extend(directory.glob(pattern))

        return sorted(files)

    def supports(self, file_path: str | Path) -> bool:
        """
        Check if a discovered file matches one of the supported patterns.
        """

        path = Path(file_path)

        suffixes = "".join(path.suffixes[-2:]).lower()

        if suffixes == ".csv.gz":
            return True

        return path.suffix.lower() in {
            ".csv",
            ".json",
        }
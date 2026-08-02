from __future__ import annotations

from pathlib import Path

from loaders.manifest.dataset_manifest import DatasetManifest


class DatasetValidator:
    """
    Validates a dataset before it is loaded.

    Responsibilities:
    - Validate dataset directory.
    - Validate supported file formats.
    - Ensure at least one supported file exists.
    """

    def validate(self, manifest: DatasetManifest) -> None:
        """
        Validate the dataset.
        """

        self._validate_directory(manifest.root_path)
        self._validate_supported_files(manifest)

    # ---------------------------------------------------------
    # Internal Validation
    # ---------------------------------------------------------

    def _validate_directory(self, directory: Path) -> None:

        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"{directory} is not a directory."
            )

    def _validate_supported_files(
        self,
        manifest: DatasetManifest,
    ) -> None:

        found = False

        for fmt in manifest.supported_formats:

            if fmt == "csv":
                if list(manifest.root_path.rglob("*.csv")):
                    found = True

                if list(manifest.root_path.rglob("*.csv.gz")):
                    found = True

            elif fmt == "json":
                if list(manifest.root_path.rglob("*.json")):
                    found = True

        if not found:
            raise FileNotFoundError(
                "No supported dataset files were found."
            )
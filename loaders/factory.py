from __future__ import annotations

from pathlib import Path

from loaders.catalog.dataset_catalog import DatasetCatalog
from loaders.manifest.dataset_manifest import DatasetManifest
from loaders.readers.csv_reader import CsvReader
from loaders.readers.json_reader import JsonReader
from loaders.registry import LoaderRegistry
from loaders.validator.dataset_validator import DatasetValidator


class LoaderFactory:
    """
    Creates fully configured dataset loaders.

    The factory is responsible for dependency construction only.
    """

    @staticmethod
    def create(
        dataset_type: str,
        dataset_path: str | Path,
    ):
        dataset_path = Path(dataset_path)
        loader_config = LoaderRegistry.get(dataset_type)

        manifest = DatasetManifest(
            dataset_name=dataset_path.name,
            dataset_type=dataset_type,
            root_path=dataset_path,
            supported_formats=list(loader_config.supported_formats),
        )

        validator = DatasetValidator()

        discovery = loader_config.discovery()

        catalog = DatasetCatalog()

        readers = {
            "csv": CsvReader(),
            "json": JsonReader(),
        }

        return loader_config.loader(
            manifest=manifest,
            validator=validator,
            discovery=discovery,
            readers=readers,
            catalog=catalog,
        )

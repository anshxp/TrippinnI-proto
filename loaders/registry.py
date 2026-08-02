from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from loaders.discovery.flat_discovery import FlatDiscovery
from loaders.discovery.recursive_discovery import RecursiveDiscovery
from loaders.mimic_loader import MimicLoader
from loaders.synthea_loader import SyntheaLoader


@dataclass(frozen=True)
class LoaderConfig:
    """
    Configuration describing a dataset loader.
    """

    loader: Type
    discovery: Type
    supported_formats: tuple[str, ...]


class LoaderRegistry:
    """
    Registry for all available dataset loaders.
    """

    LOADERS = {
        "synthea": LoaderConfig(
            loader=SyntheaLoader,
            discovery=FlatDiscovery,
            supported_formats=("csv",),
        ),

        "mimic": LoaderConfig(
            loader=MimicLoader,
            discovery=RecursiveDiscovery,
            supported_formats=("csv",),
        ),
    }

    @classmethod
    def get(cls, dataset_type: str) -> LoaderConfig:

        dataset_type = dataset_type.lower()

        if dataset_type not in cls.LOADERS:
            raise ValueError(
                f"Unsupported dataset type: {dataset_type}"
            )

        return cls.LOADERS[dataset_type]

    @classmethod
    def supported_datasets(cls) -> list[str]:
        return sorted(cls.LOADERS.keys())

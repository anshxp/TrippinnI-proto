"""
registry.py

Maps dataset types to their corresponding loaders.
"""

from loaders.synthea_loader import SyntheaLoader


class LoaderRegistry:
    """
    Registry for all supported dataset loaders.
    """

    LOADERS = {
        "synthea": SyntheaLoader,
    }

    @classmethod
    def get_loader(cls, dataset_type: str):

        if dataset_type not in cls.LOADERS:
            raise ValueError(
                f"Unsupported dataset type: {dataset_type}"
            )

        return cls.LOADERS[dataset_type]
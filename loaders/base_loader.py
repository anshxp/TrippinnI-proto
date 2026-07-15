"""
base_loader.py

Abstract interface for all dataset loaders used in TrippinnI.

Every healthcare dataset loader should inherit from BaseLoader
and implement the methods defined here.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load all dataset files into memory.
        """
        pass

    @abstractmethod
    def get_tables(self) -> list[str]:
        """
        Returns the names of all loaded tables.
        """
        pass

    @abstractmethod
    def get_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Returns a dataframe for the requested table.
        """
        pass

    @abstractmethod
    def get_schema(self) -> dict:
        """
        Returns metadata describing every table.
        """
        pass
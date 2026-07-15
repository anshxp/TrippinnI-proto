"""
loader.py

Responsible for creating the correct dataset loader.
"""

from loaders.registry import LoaderRegistry


class LoaderManager:

    def __init__(self):

        self.loader = None

    def initialize(self, dataset_type, dataset_path):

        loader_class = LoaderRegistry.get_loader(dataset_type)

        self.loader = loader_class(dataset_path)

        self.loader.load()

        return self.loader

    def get_loader(self):

        return self.loader
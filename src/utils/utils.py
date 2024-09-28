import random
from ast import List
from typing import Tuple

from rdflib import Dataset
from torch import nn


class BabyJoeyUtil:
    def __init__(self) -> None:
        pass

    @staticmethod
    def count_params(model: nn.Module) -> int:
        """Count the total number of model parameters.

        Args:
            model (nn.Module): Input model

        Returns:
            int: Total number of parameters
        """
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    @staticmethod
    def sample_dataset(dataset: Dataset, ratio: float = 0.2, seed: int = 42) -> Dataset:
        """Sample a fraction of the dataset.

        Args:
            dataset (Dataset): Dataset to sample from.
            ratio (float, optional): Sample ratio. Defaults to 0.2.
            seed (int, optional): Random seed for sampling. Defaults to 42.

        Returns:
            Dataset: Sampled dataset.
        """
        random.seed(seed)
        dataset_size = len(dataset)
        reduced_size = int(dataset_size * ratio)
        indices = random.sample(range(dataset_size), reduced_size)
        return dataset.select(indices)

import random
from torch import nn
from torch.utils.data import Dataset, Subset


class BabyJoeyUtil:
    def __init__(self) -> None:
        pass
    
    def count_params(model: nn.Module) -> int:
        """Count the total number of model parameters.

        Args:
            model (nn.Module): Input model

        Returns:
            int: Total number of parameters
        """
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    def sample_dataset(dataset: Dataset, fraction: float = 0.2) -> Subset:
        """Sample a fraction of the dataset.

        Args:
            dataset (Dataset): Dataset to sample from.
            fraction (float, optional): Sample ratio. Defaults to 0.2.

        Returns:
            Subset: Sampled dataset.
        """
        dataset_size = len(dataset)
        reduced_size = int(dataset_size * fraction)
        indices = random.sample(range(dataset_size), reduced_size)
        return Subset(dataset, indices)

    
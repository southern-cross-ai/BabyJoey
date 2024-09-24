from torch import nn


def count_parameters(model: nn.Module) -> int:  # TODO: Moved from main.py
    """Count the total number of model parameters.

    Args:
        model (nn.Module): Input model

    Returns:
        int: Total number of parameters
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

from torch import nn


def count_params(model: nn.Module) -> int:
    """Count the total number of model parameters.

    Args:
        model (nn.Module): Input model

    Returns:
        int: Total number of parameters
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

import torch.nn.functional as F
import torch


def nll_loss(output: torch.tensor, target: torch.tensor):
    """
    A softmax activation is required at the final layer of the network
    """
    return F.nll_loss(output, target)


def cse(output: torch.tensor, target: torch.tensor):
    """
    With pytorch cross entropy loss a softmax activation is NOT required
    at the final layer of network.
    """
    return F.cross_entropy(output, target)


def mse(output: torch.tensor, target: torch.tensor):
    return F.mse_loss(output, target)

from omegaconf import DictConfig
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint_sequential
import torch

class BandsplitModuleRegistry(type):
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    MODEL_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.MODEL_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.MODEL_REGISTRY)


class BaseRegisteredBandsplitModule(nn.Module, metaclass=BandsplitModuleRegistry):
    def __init__(self, *, config: DictConfig):
        super().__init__()

        self.config = config


class NormFC(nn.Module):
    """
    Fully connected layer with optional weight normalization and layer normalization.

    Args:
        emb_dim (int): Output embedding dimension.
        bandwidth (int): Bandwidth of the subband.
        in_channels (int): Number of input channels.
        use_weight_norm (bool, optional): Whether to apply weight normalization. Defaults to True.
        _verbose (bool): Verbose mode.
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        bandwidth: int,
        in_channels: int,
    ) -> None:
        super().__init__()

        reim: int = 2

        norm_in: int = in_channels * bandwidth * reim
        fc_in: int = bandwidth * reim * in_channels

        self.combined = nn.Sequential(
            nn.LayerNorm(norm_in),
            weight_norm(nn.Linear(fc_in, emb_dim)),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NormFC.

        Args:
            xb (torch.Tensor): Subband spectrogram. Shape: (batch, n_time, in_chan * bw * 2)

        Returns:
            torch.Tensor: Subband embedding. Shape: (batch, n_time, emb_dim)
        """
        return checkpoint_sequential(self.combined, 1, xb, use_reentrant=False)

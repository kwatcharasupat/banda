from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn.modules import activation
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn.utils.parametrizations import weight_norm

from banda.modules.bandsplit.band_specs import band_widths_from_specs

import structlog

logger = structlog.get_logger(__name__)

class BaseNormMLP(nn.Module):
    """
    Base class for a normalized MLP module.

    Args:
        emb_dim (int): Dimension of the input embedding.
        mlp_dim (int): Dimension of the MLP hidden layer.
        bandwidth (int): Bandwidth of the subband.
        in_channels (Optional[int]): Number of input channels.
        hidden_activation (str): Activation function for the hidden layer. Default: "Tanh".
        hidden_activation_kwargs (Optional[Dict]): Additional arguments for the activation function. Default: None.
        complex_mask (bool): Whether to use a complex mask. Default: True.
        use_weight_norm (bool): Whether to apply weight normalization. Default: True.
         (bool): Whether to enable verbose logging. Default: False.
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.hidden_activation = activation.__dict__[hidden_activation]

        self.bandwidth = bandwidth
        self.in_channels = in_channels

        self.reim = 2 
        self.glu_mult = 2

        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim

    def _make_norm(self) -> nn.Module:
        """
        Create a normalization layer.

        Returns:
            nn.Module: Normalization layer.
        """
        return nn.LayerNorm(self.emb_dim)

    def _make_hidden(
        self,
    ) -> nn.Module:
        """
        Returns:
            nn.Module: Hidden layer.
        """
        return nn.Sequential(
            weight_norm(nn.Linear(in_features=self.emb_dim, out_features=self.mlp_dim)),
            self.hidden_activation(**self.hidden_activation_kwargs),
        )

    def _make_output(self) -> nn.Module:
        """
        Create an output layer.

        Args:
            mlp_dim (int): Dimension of the MLP hidden layer.

        Returns:
            nn.Module: Output layer.
        """


        return nn.Sequential(
            weight_norm(
                nn.Linear(
                    in_features=self.mlp_dim,
                    out_features=self.bandwidth * self.in_channels * self.reim * 2,
                )
            ),
            nn.GLU(dim=-1),
        )

    def _make_combined(
        self,
    ) -> nn.Module:
        """
        Create the combined module.

        Returns:
            nn.Module: Combined module.
        """

        norm = self._make_norm()
        hidden = self._make_hidden(
        )
        output = self._make_output(
        )

        return nn.Sequential(
            norm,
            hidden,
            output,
        )


class NormMLP(BaseNormMLP):
    """
    Normalized MLP module for subband mask estimation.

    Args:
        emb_dim (int): Dimension of the input embedding.
        mlp_dim (int): Dimension of the MLP hidden layer.
        bandwidth (int): Bandwidth of the subband.
        in_channels (Optional[int]): Number of input channels.
        hidden_activation (str): Activation function for the hidden layer. Default: "Tanh".
        hidden_activation_kwargs (Optional[Dict]): Additional arguments for the activation function. Default: None.
        complex_mask (bool): Whether to use a complex mask. Default: True.
        use_weight_norm (bool): Whether to apply weight normalization. Default: True.
         (bool): Whether to enable verbose logging. Default: False.
    """

    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channels=in_channels,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
        )

        self.combined = self._make_combined(        )

    def reshape_output(self, mb: torch.Tensor) -> torch.Tensor:
        """
        Reshape the flattened subband mask.

        Args:
            mb (torch.Tensor): Flattened subband mask. Shape: (batch, n_time, bandwidth * in_channels * reim)

        Returns:
            torch.Tensor: Reshaped subband mask. Shape: (batch, in_channels, bandwidth, n_time)
        """

        batch, n_time, _ = mb.shape

        logger.info("shape", shape=mb.shape, dtype=mb.dtype, in_channels=self.in_channels, bandwidth=self.bandwidth, reim=self.reim)

        mb = mb.reshape(
            batch, n_time, self.in_channels, self.bandwidth, self.reim
        ).contiguous()
        mb = torch.view_as_complex(mb)

        mb = torch.permute(mb, (0, 2, 3, 1))  # (batch, in_channels, bandwidth, n_time)

        return mb

    def forward(self, qb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for subband mask estimation.

        Args:
            qb (torch.Tensor): Subband embedding. Shape: (batch, n_time, emb_dim)

        Returns:
            torch.Tensor: Subband mask. Shape: (batch, in_channels, bandwidth, n_time)
        """

        mb = checkpoint_sequential(self.combined, 2, qb, use_reentrant=False)
        # (batch, n_time, bandwidth * in_channels * reim)
        mb = self.reshape_output(mb)
        # (batch, in_channels, bandwidth, n_time)

        return mb


class BaseMaskEstimationModule(nn.Module):
    """
    Base class for mask estimation modules.

    Args:
        band_specs (List[Tuple[float, float]]): List of band specifications.
        emb_dim (int): Dimension of the input embedding.
        mlp_dim (int): Dimension of the MLP hidden layer.
        in_channels (Optional[int]): Number of input channels.
        hidden_activation (str): Activation function for the hidden layer. Default: "Tanh".
        hidden_activation_kwargs (Optional[Dict]): Additional arguments for the activation function. Default: None.
        complex_mask (bool): Whether to use a complex mask. Default: True.
        norm_mlp_cls (Type[nn.Module]): Class for the normalized MLP. Default: NormMLP.
        norm_mlp_kwargs (Optional[Dict]): Additional arguments for the normalized MLP. Default: None.
         (bool): Whether to enable verbose logging. Default: False.
    """

    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.band_widths = band_widths_from_specs(band_specs=band_specs)
        self.n_bands = len(band_specs)

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        if norm_mlp_kwargs is None:
            norm_mlp_kwargs = {}

        self.norm_mlp = nn.ModuleList(
            [
                norm_mlp_cls(
                    bandwidth=self.band_widths[b],
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    in_channels=in_channels,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    **norm_mlp_kwargs,
                )
                for b in range(self.n_bands)
            ]
        )

    def compute_masks(self, q: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute subband masks.

        Args:
            q (torch.Tensor): Bottleneck embedding. Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            List[torch.Tensor]: List of subband masks. Shape: [n_bands * (batch, in_channels, bandwidth, n_time)]
        """

        masks = []

        for b, nmlp in enumerate(self.norm_mlp):
            qb = q[:, b, :, :]  # (batch, n_time, emb_dim)
            mb = nmlp(qb)  # (batch, in_channels, bw, n_time)

            # safeguard against NaN
            if torch.any(torch.isnan(mb)):
                mb = torch.view_as_complex(
                    torch.nan_to_num(torch.view_as_real(mb), nan=0.0)
                )

            masks.append(mb)

        return masks


class OverlappingMaskEstimationModule(BaseMaskEstimationModule):
    """
    Mask estimation module for overlapping subbands.

    Args:
        in_channels (int): Number of input channels.
        band_specs (List[Tuple[float, float]]): List of band specifications.
        freq_weights (List[torch.Tensor]): Frequency weights for each band.
        n_freq (int): Total number of frequencies.
        emb_dim (int): Dimension of the input embedding.
        mlp_dim (int): Dimension of the MLP hidden layer.
        hidden_activation (str): Activation function for the hidden layer. Default: "Tanh".
        hidden_activation_kwargs (Optional[Dict]): Additional arguments for the activation function. Default: None.
        complex_mask (bool): Whether to use a complex mask. Default: True.
        norm_mlp_cls (Type[nn.Module]): Class for the normalized MLP. Default: NormMLP.
        norm_mlp_kwargs (Optional[Dict]): Additional arguments for the normalized MLP. Default: None.
        use_freq_weights (bool): Whether to use frequency weights. Default: True.
         (bool): Whether to enable verbose logging. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        band_specs: List[Tuple[float, float]],
        n_freq: int,
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Optional[Dict] = None,
    ) -> None:

        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channels=in_channels,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channels = in_channels

    def forward(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for overlapping subband mask estimation.

        Args:
            q (torch.Tensor): Bottleneck embedding. Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            torch.Tensor: Fullband mask. Shape: (batch, in_channels, n_freq, n_time)
        """

        batch, n_bands, n_time, _ = q.shape

        mask_list = self.compute_masks(q)
        # [n_bands * (batch, in_channels, bw, n_time)]

        masks = torch.zeros(
            (batch, self.in_channels, self.n_freq, n_time),
            device=q.device,
            dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            masks[:, :, fstart:fend, :] += mask

        return masks


# class DisjointMaskEstimationModule(OverlappingMaskEstimationModule):
#     """
#     Mask estimation module for disjoint subbands.

#     Args:
#         band_specs (List[Tuple[float, float]]): List of band specifications.
#         emb_dim (int): Dimension of the input embedding.
#         mlp_dim (int): Dimension of the MLP hidden layer.
#         in_channels (Optional[int]): Number of input channels.
#         hidden_activation (str): Activation function for the hidden layer. Default: "Tanh".
#         hidden_activation_kwargs (Optional[Dict]): Additional arguments for the activation function. Default: None.
#         complex_mask (bool): Whether to use a complex mask. Default: True.
#          (bool): Whether to enable verbose logging. Default: False.
#     """

#     def __init__(
#         self,
#         band_specs: List[Tuple[float, float]],
#         emb_dim: int,
#         mlp_dim: int,
#         in_channels: Optional[int],
#         hidden_activation: str = "Tanh",
#         hidden_activation_kwargs: Optional[Dict] = None,
#         complex_mask: bool = True,
#         : bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             in_channels=in_channels,
#             band_specs=band_specs,
#             freq_weights=None,
#             n_freq=None,
#             emb_dim=emb_dim,
#             mlp_dim=mlp_dim,
#             hidden_activation=hidden_activation,
#             hidden_activation_kwargs=hidden_activation_kwargs,
#             complex_mask=complex_mask,
#             =,
#         )

#     def forward(
#         self,
#         q: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Forward pass for disjoint subband mask estimation.

#         Args:
#             q (torch.Tensor): Bottleneck embedding. Shape: (batch, n_bands, n_time, emb_dim)

#         Returns:
#             torch.Tensor: Fullband mask. Shape: (batch, in_channels, n_freq, n_time)
#         """

#         masks = self.compute_masks(q)
#         # [n_bands * (batch, in_channels, bandwidth, n_time)]

#         masks = torch.concat(masks, dim=2)
#         # (batch, in_channels, n_freq, n_time)

#         return masks

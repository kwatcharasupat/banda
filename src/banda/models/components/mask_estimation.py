#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import warnings
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation

from banda.models.utils import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class BaseNormMLP(nn.Module):
    """
    Base class for Normalized Multi-Layer Perceptron (MLP) used in mask estimation.
    """
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__()
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=mlp_dim),
            activation.__dict__[hidden_activation](
                **self.hidden_activation_kwargs
            ),
        )

        self.bandwidth = bandwidth
        self.in_channel = in_channel

        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.glu_mult = 2 # For GLU output, it's typically 2x the desired output size


class NormMLP(BaseNormMLP):
    """
    Normalized MLP for mask estimation, producing a single mask.
    """
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )

        self.output = nn.Sequential(
            nn.Linear(
                in_features=mlp_dim,
                out_features=bandwidth * in_channel * self.reim * self.glu_mult,
            ),
            nn.GLU(dim=-1),
        )

    def reshape_output(self, mb: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the output of the MLP to the mask format.
        """
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(
                batch,
                n_time,
                self.in_channel,
                self.bandwidth,
                self.reim
            ).contiguous()
            mb = torch.view_as_complex(mb)  # (batch, n_time, in_channel, bandwidth)
        else:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth)

        mb = torch.permute(
            mb,
            (0, 2, 3, 1) # (batch, in_channel, bandwidth, n_time)
        )
        return mb

    def forward(self, qb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NormMLP.

        Args:
            qb (torch.Tensor): Input tensor from the time-frequency model.
                               Shape: (batch, n_time, emb_dim)

        Returns:
            torch.Tensor: Predicted mask. Shape: (batch, in_channel, bandwidth, n_time)
        """
        qb = self.norm(qb)
        qb = self.hidden(qb)
        mb = self.output(qb)
        mb = self.reshape_output(mb)
        return mb


class MultAddNormMLP(NormMLP):
    """
    Normalized MLP for mask estimation, producing both multiplicative and additive masks.
    """
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(emb_dim, mlp_dim, bandwidth, in_channel, hidden_activation, hidden_activation_kwargs, complex_mask)

        self.output2 = nn.Sequential(
            nn.Linear(
                in_features=mlp_dim,
                out_features=bandwidth * in_channel * self.reim * self.glu_mult,
            ),
            nn.GLU(dim=-1),
        )

    def forward(self, qb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MultAddNormMLP.

        Args:
            qb (torch.Tensor): Input tensor from the time-frequency model.
                               Shape: (batch, n_time, emb_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted multiplicative and additive masks.
                                               Each tensor has shape: (batch, in_channel, bandwidth, n_time)
        """
        qb = self.norm(qb)
        qb = self.hidden(qb)
        mmb = self.output(qb)
        mmb = self.reshape_output(mmb)
        amb = self.output2(qb)
        amb = self.reshape_output(amb)
        return mmb, amb


class MaskEstimationModuleSuperBase(nn.Module):
    """
    Super base class for mask estimation modules.
    """
    pass


class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    """
    Base class for mask estimation modules.
    """
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            norm_mlp_cls: Type[nn.Module] = NormMLP,
            norm_mlp_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.band_widths = band_widths_from_specs(band_specs)
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
                    in_channel=in_channel,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    complex_mask=complex_mask,
                    **norm_mlp_kwargs,
                )
                for b in range(self.n_bands)
            ]
        )

    def compute_masks(self, q: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes masks for each band.

        Args:
            q (torch.Tensor): Input tensor from the time-frequency model.
                              Shape: (batch, n_bands, n_time, emb_dim)

        Returns:
            List[torch.Tensor]: A list of predicted masks for each band.
                                Each mask tensor has shape: (batch, in_channel, bandwidth, n_time)
        """
        batch, n_bands, n_time, emb_dim = q.shape

        masks = []

        for b, nmlp in enumerate(self.norm_mlp):
            qb = q[:, b, :, :] # (batch, n_time, emb_dim)
            mb = nmlp(qb)
            masks.append(mb)

        return masks


class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    """
    Mask estimation module for overlapping frequency bands.
    """
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            freq_weights: Optional[List[torch.Tensor]],
            n_freq: int,
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            cond_dim: int = 0,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            norm_mlp_cls: Type[nn.Module] = NormMLP,
            norm_mlp_kwargs: Optional[Dict] = None,
            use_freq_weights: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs) # Overlapping bands can still have no gaps

        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim + cond_dim, # Add cond_dim to emb_dim if conditioning is used
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channel = in_channel

        if freq_weights is not None:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights_{i}", fw) # Use register_buffer for non-trainable tensors
            self.use_freq_weights = use_freq_weights
        else:
            self.use_freq_weights = False

        self.cond_dim = cond_dim

    def forward(self, q: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the OverlappingMaskEstimationModule.

        Args:
            q (torch.Tensor): Input tensor from the time-frequency model.
                              Shape: (batch, n_bands, n_time, emb_dim)
            cond (Optional[torch.Tensor]): Conditioning tensor.

        Returns:
            torch.Tensor: Predicted full-band mask. Shape: (batch, in_channel, n_freq, n_time)
        """
        batch, n_bands, n_time, emb_dim = q.shape

        if cond is not None:
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time, "Conditioning tensor time dimension must match."
                cond = cond[:, None, :, :].expand(-1, n_bands, -1, -1) # Expand to n_bands
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")
            q = torch.cat([q, cond], dim=-1)
        elif self.cond_dim > 0:
            # If cond_dim is set but no conditioning tensor is provided,
            # create a dummy one (e.g., all ones)
            warnings.warn("cond_dim > 0 but no conditioning tensor provided. Using dummy conditioning.")
            cond = torch.ones(
                (batch, n_bands, n_time, self.cond_dim),
                device=q.device,
                dtype=q.dtype,
            )
            q = torch.cat([q, cond], dim=-1)

        mask_list = self.compute_masks(q) # [n_bands * (batch, in_channel, bandwidth, n_time)]

        masks = torch.zeros(
            (batch, self.in_channel, self.n_freq, n_time),
            device=q.device,
            dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            if self.use_freq_weights and hasattr(self, f"freq_weights_{im}"):
                fw = getattr(self, f"freq_weights_{im}")[:, None, None] # (bandwidth, 1, 1)
                mask = mask * fw
            masks[:, :, fstart:fend, :] += mask

        return masks


class MaskEstimationModule(OverlappingMaskEstimationModule):
    """
    Mask estimation module for non-overlapping frequency bands.
    """
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            cond_dim: int = 0, # Added cond_dim for consistency
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            **kwargs, # Catch any extra kwargs from OverlappingMaskEstimationModule
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs) # Crucial for non-overlapping bands
        super().__init__(
            band_specs=band_specs,
            freq_weights=None, # No frequency weights for non-overlapping
            n_freq=sum(band_widths_from_specs(band_specs)), # Total frequency bins
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            cond_dim=cond_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_freq_weights=False, # Explicitly set to False
            **kwargs,
        )

    def forward(self, q: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the MaskEstimationModule.

        Args:
            q (torch.Tensor): Input tensor from the time-frequency model.
                              Shape: (batch, n_bands, n_time, emb_dim)
            cond (Optional[torch.Tensor]): Conditioning tensor.

        Returns:
            torch.Tensor: Predicted full-band mask. Shape: (batch, in_channel, n_freq, n_time)
        """
        batch, n_bands, n_time, emb_dim = q.shape

        if cond is not None:
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time, "Conditioning tensor time dimension must match."
                cond = cond[:, None, :, :].expand(-1, n_bands, -1, -1) # Expand to n_bands
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")
            q = torch.cat([q, cond], dim=-1)
        elif self.cond_dim > 0:
            warnings.warn("cond_dim > 0 but no conditioning tensor provided. Using dummy conditioning.")
            cond = torch.ones(
                (batch, n_bands, n_time, self.cond_dim),
                device=q.device,
                dtype=q.dtype,
            )
            q = torch.cat([q, cond], dim=-1)

        mask_list = self.compute_masks(q) # [n_bands * (batch, in_channel, bandwidth, n_time)]

        # For non-overlapping bands, we can simply concatenate the masks along the frequency dimension
        masks = torch.cat(
            masks,
            dim=2 # Concatenate along the frequency dimension
        ) # (batch, in_channel, n_freq, n_time)

        return masks
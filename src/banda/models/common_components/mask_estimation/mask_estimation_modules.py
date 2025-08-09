#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import warnings
from typing import Dict, List, Optional, Tuple, Type, Union # Added Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation
from torch.nn.utils import weight_norm
import structlog
import logging
from omegaconf import DictConfig, OmegaConf
import hydra.utils # Added import

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

from banda.models.common_components.configs.mask_estimation_configs import MaskEstimationConfig # Import MaskEstimationConfig


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
            weight_norm(nn.Linear(in_features=emb_dim, out_features=mlp_dim)),
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
        self.output_linear = weight_norm(nn.Linear(
            in_features=mlp_dim,
            out_features=bandwidth * in_channel * self.reim * self.glu_mult,
        ))
        self.output_glu = nn.GLU(dim=-1)

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
                                (batch, n_time, emb_dim)

        Returns:
            torch.Tensor: Predicted mask. Shape: (batch, in_channel, bandwidth, n_time)
        """
        logger.debug("NormMLP: qb before normalization", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.mean().item())
        qb = self.norm(qb)
        if torch.isnan(qb).any():
            logger.error("NormMLP: NaN detected in qb after normalization", mean_val=qb.mean().item())
            raise ValueError("NaN in qb after normalization")
        logger.debug("NormMLP: qb after normalization", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        qb = self.hidden(qb)
        if torch.isnan(qb).any():
            logger.error("NormMLP: NaN detected in qb after hidden layer", mean_val=qb.mean().item())
            raise ValueError("NaN in qb after hidden layer")
        logger.debug("NormMLP: qb after hidden layer", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        
        # If the standard deviation is very small, add a small amount of noise to prevent NaNs
        std_dev = qb.std(dim=-1, keepdim=True)
        if (std_dev < 1e-6).any(): # Check for very small standard deviation
            logger.warning(f"NormMLP: Very small standard deviation detected in qb before output_linear. Adding noise.")
            qb = qb + torch.randn_like(qb) * 1e-6
        logger.debug("NormMLP: qb before output_linear", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        logger.debug(f"NormMLP: qb shape before output_linear: {qb.shape}") # ADDED THIS LINE
        linear_out = self.output_linear(qb)
        if torch.isnan(linear_out).any():
            logger.error("NormMLP: NaN detected in linear_out after output_linear", mean_val=linear_out.mean().item())
            raise ValueError("NaN in linear_out after output_linear")
        logger.debug("NormMLP: linear_out after output_linear", mean_val=linear_out.mean().item(), min_val=linear_out.min().item(), max_val=linear_out.max().item())

        logger.debug("NormMLP: linear_out before output_glu", mean_val=linear_out.mean().item(), min_val=linear_out.min().item(), max_val=linear_out.max().item())
        mb = self.output_glu(linear_out)
        if torch.isnan(mb).any():
            logger.error("NormMLP: NaN detected in mb after output_glu", mean_val=mb.mean().item())
            raise ValueError("NaN in mb after output_glu")
        logger.debug("NormMLP: mb after output_glu", mean_val=mb.mean().item(), min_val=mb.min().item(), max_val=mb.max().item())
        
        mb = self.reshape_output(mb)
        if torch.isnan(mb).any():
            logger.error("NormMLP: NaN detected in mb after reshape_output", mean_val=mb.mean().item())
            raise ValueError("NaN in mb after reshape_output")
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

        self.output2_linear = weight_norm(nn.Linear(
            in_features=mlp_dim,
            out_features=bandwidth * in_channel * self.reim * self.glu_mult,
        ))
        self.output2_glu = nn.GLU(dim=-1)

    def forward(self, qb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MultAddNormMLP.

        Args:
            qb (torch.Tensor): Input tensor from the time-frequency model.
                                (batch, n_time, emb_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted multiplicative and additive masks.
                                               Each tensor has shape: (batch, in_channel, bandwidth, n_time)
        """
        logger.debug("MultAddNormMLP: qb before normalization", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        qb = self.norm(qb)
        if torch.isnan(qb).any():
            logger.error("MultAddNormMLP: NaN detected in qb after normalization", mean_val=qb.mean().item())
            raise ValueError("NaN in qb after normalization")
        logger.debug("MultAddNormMLP: qb after normalization", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        qb = self.hidden(qb)
        if torch.isnan(qb).any():
            logger.error("MultAddNormMLP: NaN detected in qb after hidden layer", mean_val=qb.mean().item())
            raise ValueError("NaN in qb after hidden layer")
        logger.debug("MultAddNormMLP: qb after hidden layer", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        
        logger.debug("MultAddNormMLP: qb before output_linear", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        linear_out_mmb = self.output_linear(qb)
        if torch.isnan(linear_out_mmb).any():
            logger.error("MultAddNormMLP: NaN detected in linear_out_mmb after output_linear", mean_val=linear_out_mmb.mean().item())
            raise ValueError("NaN in linear_out_mmb after output_linear")
        logger.debug("MultAddNormMLP: linear_out_mmb after output_linear", mean_val=linear_out_mmb.mean().item(), min_val=linear_out_mmb.min().item(), max_val=linear_out_mmb.max().item())

        logger.debug("MultAddNormMLP: linear_out_mmb before output_glu", mean_val=linear_out_mmb.mean().item(), min_val=linear_out_mmb.min().item(), max_val=linear_out_mmb.max().item())
        mmb = self.output_glu(linear_out_mmb)
        if torch.isnan(mmb).any():
            logger.error("MultAddNormMLP: NaN detected in mmb after output_glu", mean_val=mmb.mean().item())
            raise ValueError("NaN in mmb after output_glu")
        logger.debug("MultAddNormMLP: mmb after output_glu", mean_val=mmb.mean().item(), min_val=mmb.min().item(), max_val=mmb.max().item())
        
        mmb = self.reshape_output(mmb)
        if torch.isnan(mmb).any():
            logger.error("MultAddNormMLP: NaN detected in mmb after reshape_output", mean_val=mmb.mean().item())
            raise ValueError("NaN in mmb after reshape_output")

        logger.debug("MultAddNormMLP: qb before output2_linear", mean_val=qb.mean().item(), min_val=qb.min().item(), max_val=qb.max().item())
        linear_out_amb = self.output2_linear(qb)
        if torch.isnan(linear_out_amb).any():
            logger.error("MultAddNormMLP: NaN detected in linear_out_amb after output2_linear", mean_val=linear_out_amb.mean().item())
            raise ValueError("NaN in linear_out_amb after output2_linear")
        logger.debug("MultAddNormMLP: linear_out_amb after output2_linear", mean_val=linear_out_amb.mean().item(), min_val=linear_out_amb.min().item(), max_val=linear_out_amb.max().item())

        logger.debug("MultAddNormMLP: linear_out_amb before output2_glu", mean_val=linear_out_amb.mean().item(), min_val=linear_out_amb.min().item(), max_val=linear_out_amb.max().item())
        amb = self.output2_glu(linear_out_amb)
        if torch.isnan(amb).any():
            logger.error("MultAddNormMLP: NaN detected in amb after output2_glu", mean_val=amb.mean().item())
            raise ValueError("NaN in amb after output2_glu")
        logger.debug("MultAddNormMLP: amb after output2_glu", mean_val=amb.mean().item(), min_val=amb.min().item(), max_val=amb.max().item())
        
        amb = self.reshape_output(amb)
        if torch.isnan(amb).any():
            logger.error("MultAddNormMLP: NaN detected in amb after reshape_output", mean_val=amb.mean().item())
            raise ValueError("NaN in amb after reshape_output")

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
            band_specs: List[Tuple[float, float]], # Added band_specs
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

        self.band_widths = [fend - fstart for fstart, fend in band_specs]
        self.n_bands = len(band_specs)
        self.band_specs = band_specs # Store band_specs

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
                                (batch, n_bands, n_time, emb_dim)

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
            # freq_weights: Optional[List[torch.Tensor]], # Deprecated
            n_freq: int,
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            norm_mlp_cls: Type[nn.Module] = NormMLP,
            norm_mlp_kwargs: Optional[Dict] = None,
            # use_freq_weights: bool = True, # Deprecated
    ) -> None:

        super().__init__(
            band_specs=band_specs, # Pass band_specs to super
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        # self.band_specs = band_specs # Already stored in super().__init__
        self.in_channel = in_channel
        self.use_freq_weights = False # Explicitly set to False as it's deprecated

    def forward(self, q: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the OverlappingMaskEstimationModule.

        Args:
            q (torch.Tensor): Input tensor from the time-frequency model.
                                (batch, n_bands, n_time, emb_dim)
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

        mask_list = self.compute_masks(q) # [n_bands * (batch, in_channel, bandwidth, n_time)]
        
        
        
        if any(torch.isnan(m).any() for m in mask_list):
            logger.error("OverlappingMaskEstimationModule: NaN detected in mask_list before combining", mean_vals=[m.mean().item() for m in mask_list])
            raise ValueError("NaN in mask_list in OverlappingMaskEstimationModule")
        masks = torch.zeros(
            (batch, self.in_channel, self.n_freq, n_time),
            device=q.device,
            dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            # if self.use_freq_weights and hasattr(self, f"freq_weights_{im}"): # Deprecated
            #     fw = getattr(self, f"freq_weights_{im}")[:, None, None] # (bandwidth, 1, 1)
            #     mask = mask * fw
            masks[:, :, fstart:fend, :] += mask

        return masks

    @classmethod
    def from_config(cls, cfg: MaskEstimationConfig, band_specs: List[Tuple[float, float]], n_freq: int) -> "OverlappingMaskEstimationModule": # Changed signature
        """
        Instantiates OverlappingMaskEstimationModule from a MaskEstimationConfig.
        """
        return cls(
            band_specs=band_specs, # Use passed band_specs
            n_freq=n_freq, # Use passed n_freq
            emb_dim=cfg.emb_dim,
            mlp_dim=cfg.mlp_dim,
            in_channel=cfg.in_channel,
            
            hidden_activation=cfg.hidden_activation,
            hidden_activation_kwargs=cfg.hidden_activation_kwargs,
            complex_mask=cfg.complex_mask,
            norm_mlp_cls=hydra.utils.get_class(cfg.norm_mlp_cls),
            norm_mlp_kwargs=cfg.norm_mlp_kwargs,
        )


class MaskEstimationModule(OverlappingMaskEstimationModule):
    """
    Mask estimation module for non-overlapping frequency bands.
    """
    def __init__(
            self,
            band_specs: List[Tuple[float, float]], # Added band_specs
            n_freq: int, # Added n_freq as a direct parameter
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            **kwargs, # Catch any extra kwargs from OverlappingMaskEstimationModule
    ) -> None:
        
        logger.debug(
            "MaskEstimationModule init",
            band_specs=band_specs,
            calculated_n_freq=n_freq # Use the passed n_freq
        )
        super().__init__(
            band_specs=band_specs, # Pass band_specs to super
            n_freq=n_freq, # Pass the direct n_freq
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            **kwargs,
        )

    def forward(self, q: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the MaskEstimationModule.
        Args:
            q (torch.Tensor): Input tensor from the time-frequency model.
                                (batch, n_bands, n_time, emb_dim)
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

        mask_list = self.compute_masks(q) # [n_bands * (batch, in_channel, bandwidth, n_time)]

        # For non-overlapping bands, we can simply concatenate the masks along the frequency dimension
        if any(torch.isnan(m).any() for m in mask_list):
            logger.error("MaskEstimationModule: NaN detected in mask_list before concatenating", mean_vals=[m.mean().item() for m in mask_list])
            raise ValueError("NaN in mask_list in MaskEstimationModule")
        masks = torch.cat(mask_list,
            dim=2 # Concatenate along the frequency dimension
        ) # (batch, in_channel, n_freq, n_time)

        return masks

    @classmethod
    def from_config(cls, cfg: MaskEstimationConfig, band_specs: List[Tuple[float, float]], n_freq: int) -> "MaskEstimationModule": # Changed signature
        """
        Instantiates MaskEstimationModule from a MaskEstimationConfig.
        """
        return cls(
            band_specs=band_specs, # Use passed band_specs
            n_freq=n_freq, # Use passed n_freq
            emb_dim=cfg.emb_dim,
            mlp_dim=cfg.mlp_dim,
            in_channel=cfg.in_channel,
            
            hidden_activation=cfg.hidden_activation,
            hidden_activation_kwargs=cfg.hidden_activation_kwargs,
            complex_mask=cfg.complex_mask,
        )
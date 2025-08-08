#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Type
from omegaconf import DictConfig, OmegaConf
import hydra.utils
import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

from banda.core.interfaces import BaseModel
from banda.models.common_components.spectral_components.spectral_base import BandsplitSpecification, get_bandsplit_specs_factory, SpectralComponent
from banda.models.common_components.bandsplit.band_split_module import BandSplitModule
from banda.models.common_components.configs.common_configs import STFTConfig, BandsplitModuleConfig, BanditSeparatorConfig # BanditSeparatorConfig will be renamed to BanditBanditSeparatorConfig later

class BaseBandsplitSeparator(BaseModel, SpectralComponent):
    """
    Base class for separator models, encapsulating common STFT and Bandsplit logic.
    """
    def __init__(self, config: BanditSeparatorConfig) -> None: # Using BanditSeparatorConfig for now, will be more generic later
        BaseModel.__init__(self)
        SpectralComponent.__init__(
            self,
            n_fft=config.stft_config.n_fft,
            win_length=config.stft_config.win_length,
            hop_length=config.stft_config.hop_length,
            window_fn=config.stft_config.window_fn,
            wkwargs=config.stft_config.wkwargs,
            power=config.stft_config.power,
            center=config.stft_config.center,
            normalized=config.stft_config.normalized,
            pad_mode=config.stft_config.pad_mode,
            onesided=config.stft_config.onesided,
        )

        self.band_specs: BandsplitSpecification = config.bandsplit_config.band_specs
        self.n_fft: int = config.stft_config.n_fft
        self.in_channel: int = config.in_channel
        self.drop_dc_band: bool = True # This should eventually come from config

        if self.band_specs is None:
            raise ValueError("band_specs must be provided to BaseBandsplitSeparator.")

        try:
            self.band_split: BandSplitModule = BandSplitModule.from_config(
                cfg=config.bandsplit_config,
                stft_config=config.stft_config,
                emb_dim=config.emb_dim, # Pass emb_dim
                in_channel=config.in_channel, # Pass in_channel
                drop_dc_band=self.drop_dc_band # Pass drop_dc_band
            )
        except Exception as e:
            logger.error(f"BaseBandsplitSeparator: Error instantiating BandSplitModule: {e}", exc_info=True)
            raise # Re-raise the exception to see the full traceback


    def _process_spectrogram(self, mixture_audio: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Performs STFT, drops DC band, and prepares spectrogram for bandsplit.
        Returns normalized spectrogram for processing and original spectrogram for masking.
        """
        original_audio_length: int = mixture_audio.shape[-1]

        # Removed global Z-normalization
        # audio_mean = mixture_audio.mean()
        # audio_std = mixture_audio.std()
        # stable_std = torch.where(audio_std < 1e-8, torch.ones_like(audio_std), audio_std)
        # normalized_mixture_audio = (mixture_audio - audio_mean) / stable_std
        # logger.debug("BaseBandsplitSeparator: Global Z-normalization applied", mean=audio_mean.item(), std=audio_std.item())
        # if torch.isnan(normalized_mixture_audio).any():
        #     logger.error("BaseBandsplitSeparator: NaN detected in normalized_mixture_audio after Z-normalization", mean_val=normalized_mixture_audio.mean().item())
        #     raise ValueError("NaN in normalized_mixture_audio")

        # Perform STFT on the original audio for feature extraction (no normalization here)
        normalized_mixture_spec: torch.Tensor = SpectralComponent.forward(self, mixture_audio) # Explicitly call SpectralComponent's forward
        if torch.isnan(normalized_mixture_spec).any():
            logger.error("BaseBandsplitSeparator: NaN detected in normalized_mixture_spec after STFT", mean_val=normalized_mixture_spec.mean().item())
            raise ValueError("NaN in normalized_mixture_spec")

        # Perform STFT on the original audio for masking
        original_mixture_spec: torch.Tensor = SpectralComponent.forward(self, mixture_audio) # Explicitly call SpectralComponent's forward
        if torch.isnan(original_mixture_spec).any():
            logger.error("BaseBandsplitSeparator: NaN detected in original_mixture_spec after STFT", mean_val=original_mixture_spec.mean().item())
            raise ValueError("NaN in original_mixture_spec")

        if self.drop_dc_band:
            normalized_mixture_spec = normalized_mixture_spec[:, :, 1:, :, :]
            original_mixture_spec = original_mixture_spec[:, :, 1:, :, :]

        input_to_bandsplit: torch.Tensor = torch.cat([normalized_mixture_spec[..., 0], normalized_mixture_spec[..., 1]], dim=1)

        return input_to_bandsplit, original_audio_length, original_mixture_spec

    def _process_bandsplit(self, input_to_bandsplit: torch.Tensor) -> torch.Tensor:
        """
        Performs bandsplit on the prepared spectrogram.
        """
        band_features: torch.Tensor = self.band_split(input_to_bandsplit)
        if torch.isnan(band_features).any():
            logger.error("BaseBandsplitSeparator: NaN detected in band_features after BandSplit", mean_val=band_features.mean().item())
            raise ValueError("NaN in band_features")
        return band_features

    def _reconstruct_audio(self, masked_spec: torch.Tensor, original_audio_length: int) -> torch.Tensor:
        """
        Reconstructs audio from masked spectrogram. Denormalization is no longer needed here
        as masking is applied to the original (unnormalized) spectrogram.
        """
        if self.drop_dc_band:
            batch_size, num_channels, freq_bins, time_frames, _ = masked_spec.shape # Unpack 5 dimensions
            dc_band_zeros = torch.zeros(batch_size, num_channels, 1, time_frames, 2, # Add 5th dimension for real/imag
                                        device=masked_spec.device, dtype=masked_spec.dtype)
            masked_spec = torch.cat([dc_band_zeros, masked_spec], dim=2)

        reconstructed_audio = self.inverse(masked_spec, original_audio_length)
        if torch.isnan(reconstructed_audio).any():
            logger.error(f"BaseBandsplitSeparator: NaN detected in reconstructed_audio after ISTFT", mean_val=reconstructed_audio.mean().item())
            raise ValueError(f"NaN in reconstructed_audio")
        
        # No denormalization needed here as masking was applied to original spectrogram
        return reconstructed_audio

    @classmethod
    def _instantiate_bandsplit_config(cls, model_cfg: DictConfig, stft_config: STFTConfig) -> BandsplitModuleConfig:
        """
        Helper to instantiate BandsplitModuleConfig.
        """
        bandsplit_type = model_cfg.bandsplit.band_specs.bandsplit_type
        n_bands = model_cfg.bandsplit.band_specs.n_bands
        fs = model_cfg.get("fs", 44100)
        drop_dc_band = model_cfg.get("drop_dc_band", True)

        band_specs: BandsplitSpecification = get_bandsplit_specs_factory(
            bandsplit_type=bandsplit_type,
            n_fft=stft_config.n_fft,
            fs=fs,
            n_bands=n_bands,
            drop_dc_band=drop_dc_band
        )

        bandsplit_config: BandsplitModuleConfig = hydra.utils.instantiate(
            model_cfg.bandsplit,
            band_specs=band_specs,
            _recursive_=False
        )
        return bandsplit_config
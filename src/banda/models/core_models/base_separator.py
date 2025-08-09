#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Type
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
import hydra.utils

from banda.data.batch_types import AudioSignal
from banda.models.common_components.spectral_components.stft import STFT
from banda.models.common_components.spectral_components.bandsplit import BandsplitModule
from banda.models.common_components.mask_estimation.mask_estimation_modules import MaskEstimationModule
from banda.models.common_components.time_frequency_models.tf_models import BaseTimeFrequencyModel
from banda.models.common_components.configs.common_configs import BaseSeparatorConfig, STFTConfig, BandsplitModuleConfig, BaseTFModelConfig, MaskEstimationConfig


class BaseSeparator(nn.Module, ABC):
    """
    Base class for all separator models.

    Provides common functionalities like STFT/ISTFT, bandsplitting,
    and handling of time-frequency models and mask estimation.
    """

    def __init__(self, config: BaseSeparatorConfig) -> None:
        """
        Initializes the BaseSeparator.

        Args:
            config (BaseSeparatorConfig): A Pydantic configuration object containing
                                       all necessary parameters for the model.
        """
        super().__init__()
        self.config = config
        self.stems = config.stems
        self.fs = config.fs
        self.emb_dim = config.emb_dim
        self.in_channel = config.in_channel

        # Initialize STFT module
        if isinstance(config.stft, STFTConfig):
            self.stft_module = hydra.utils.instantiate(config.stft)
        else:
            # Assume it's already an instantiated STFT module if not a config
            self.stft_module = config.stft

        # Initialize Bandsplit module
        if isinstance(config.bandsplit, BandsplitModuleConfig):
            self.bandsplit_module = hydra.utils.instantiate(config.bandsplit)
        else:
            # Assume it's already an instantiated BandsplitModule if not a config
            self.bandsplit_module = config.bandsplit

        # Initialize Time-Frequency model
        if isinstance(config.tfmodel, DictConfig):
            self.tf_model = hydra.utils.instantiate(config.tfmodel)
        else:
            # Assume it's already an instantiated TF model if not a DictConfig
            self.tf_model = config.tfmodel

        # Initialize Mask Estimation modules
        self.mask_estim: nn.ModuleDict = nn.ModuleDict()
        for stem in self.stems:
            if isinstance(config.mask_estim, MaskEstimationConfig):
                # Pass n_freq to mask estimation module
                mask_estim_config_dict = OmegaConf.to_container(config.mask_estim, resolve=True)
                mask_estim_config_dict["n_freq"] = self.stft_module.n_fft // 2 + 1
                self.mask_estim[stem] = hydra.utils.instantiate(OmegaConf.create(mask_estim_config_dict))
            else:
                # Assume it's already an instantiated MaskEstimationModule if not a config
                self.mask_estim[stem] = config.mask_estim


    def _apply_stft(self, audio: torch.Tensor) -> AudioSignal:
        """Applies STFT to the audio."""
        spectrogram = self.stft_module(audio)
        return AudioSignal(audio=audio, spectrogram=spectrogram)

    def _apply_bandsplit(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Applies bandsplitting to the spectrogram."""
        return self.bandsplit_module(spectrogram)

    def _apply_tf_model(self, band_features: torch.Tensor) -> torch.Tensor:
        """Applies the time-frequency model."""
        return self.tf_model(band_features)

    def _apply_mask_estimation(self, processed_features: torch.Tensor) -> torch.Tensor:
        """Applies mask estimation."""
        # This method will be called for each stem in the forward pass
        # The actual mask estimation module is selected in the forward pass
        # based on the stem.
        return processed_features # Placeholder, actual mask estimation happens in forward

    def _apply_istft(self, masked_spectrogram: torch.Tensor, audio_length: int) -> torch.Tensor:
        """Applies ISTFT to the masked spectrogram."""
        return self.stft_module.inverse(masked_spectrogram, audio_length)

    @abstractmethod
    def forward(self, mixture_audio: torch.Tensor) -> Dict[str, AudioSignal]:
        """
        Abstract forward pass method.

        Must be implemented by subclasses.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: DictConfig) -> "BaseSeparator":
        """
        Abstract class method to instantiate the model from a DictConfig.

        Must be implemented by subclasses.
        """
        pass
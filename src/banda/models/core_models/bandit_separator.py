#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Type
from omegaconf import DictConfig, OmegaConf
from banda.utils.registry import MODELS_REGISTRY
import hydra.utils # Re-import hydra.utils

from banda.data.batch_types import FixedStemSeparationBatch, AudioSignal
from banda.core.interfaces import BaseModel
from banda.models.core_models.base_separator import BaseSeparator # Changed import to BaseSeparator
from banda.models.common_components.mask_estimation.mask_estimation_modules import MaskEstimationModule # Removed OverlappingMaskEstimationModule
from banda.models.common_components.utils.constants import VDBO_STEMS # Example default stems
from banda.models.common_components.configs.common_configs import STFTConfig, BandsplitModuleConfig, BaseTFModelConfig, MaskEstimationConfig, BanditSeparatorConfig
from banda.models.common_components.time_frequency_models.tf_models import RNNTFModelConfig, TransformerTFModelConfig, MambaTFModelConfig
from banda.models.common_components.spectral_components.spectral_base import BandsplitSpecification
from banda.models.common_components.configs.bandsplit_configs import VocalBandsplitSpecsConfig, MusicalBandsplitSpecsConfig # Corrected import
from banda.models.common_components.time_frequency_models.tf_models import SeqBandModellingModule # Import specific TF models


@MODELS_REGISTRY.register("bandit")

class Bandit(BaseSeparator): # Inherit from BaseSeparator
    """
    A fixed-stem Bandit-like source separation model.

    This model takes a mixture audio as input and outputs separated sources
    using a bandsplit approach with a time-frequency model and mask estimation.
    It inherits from BaseModel for general model interface and SpectralComponent
    for STFT/ISTFT functionalities.
    """

    def __init__(self, config: BanditSeparatorConfig) -> None:
        """
        Initializes the Separator model.

        Args:
            config (BanditSeparatorConfig): A Pydantic configuration object containing
                                       all necessary parameters for the model.

        Raises:
            ValueError: If `band_specs` is not provided in the configuration.
            ValueError: If an unknown `tf_model_type` is specified in the configuration.
        """
        super().__init__(config) # Initialize BaseSeparator first

        # Instantiate mask_estim using its from_config method
        # This part is already handled in BaseSeparator's __init__
        # self.mask_estim: nn.ModuleDict = nn.ModuleDict(
        #     {
        #         stem: self.mask_estim_cls.from_config(
        #             OmegaConf.create(self.mask_estim_config_params)
        #         )
        #         for stem in self.stems
        #     }
        # )

    def forward(self, mixture_audio: torch.Tensor) -> Dict[str, AudioSignal]:
        """
        Forward pass of the separator model.

        Args:
            mixture_audio (torch.Tensor): The input mixture audio tensor.
                Shape: (batch_size, channels, samples)

        Returns:
            Dict[str, AudioSignal]: Dictionary of separated source AudioSignal objects.
        """
        mixture_signal: AudioSignal = self._apply_stft(mixture_audio) # Changed from _process_spectrogram
        band_features = self._apply_bandsplit(mixture_signal.spectrogram) # Changed from _process_bandsplit

        # 3. Time-Frequency Model: Process band features
        # Output shape: (batch, n_bands, n_time, emb_dim)
        processed_features: torch.Tensor = self._apply_tf_model(band_features) # Changed from self.tf_model
        if torch.isnan(processed_features).any():
            raise ValueError("NaN in processed_features")

        # 4. Mask Estimation: Predict masks for each source
        # Output shape: Dict[stem_name, (batch, in_channel, freq, time)]
        masks: Dict[str, torch.Tensor] = {
            stem: self._apply_mask_estimation(processed_features) # Changed from self.mask_estim
            for stem in self.stems
        }

        # 5. Apply Masks and ISTFT: Reconstruct separated audio
        separated_signals: Dict[str, AudioSignal] = {}
        for stem, mask in masks.items():
            if torch.isnan(mask).any():
                raise ValueError(f"NaN in mask for stem {stem}")
            
            # Apply mask to the mixture spectrogram
            raw_mixture_spec = mixture_signal.spectrogram
            
            # Convert raw_mixture_spec to complex before multiplication
            raw_mixture_spec_complex = torch.view_as_complex(raw_mixture_spec)
            
            # mask is (batch, channels, freq, time)
            # raw_mixture_spec_complex is (batch, channels, freq, time)
            # No unsqueeze needed for mask as it already matches the dimensions for element-wise multiplication
            masked_spec_complex: torch.Tensor = torch.view_as_complex(raw_mixture_spec) * mask
            
            # Convert back to real tensor for _reconstruct_audio
            masked_spec: torch.Tensor = torch.view_as_real(masked_spec_complex)
            
            # Reconstruct audio from the masked spectrogram
            reconstructed_audio = self._apply_istft(masked_spec, mixture_audio.shape[-1]) # Changed from _reconstruct_audio_from_masked_spec
            
            # Create an AudioSignal object for the separated source
            separated_signals[stem] = AudioSignal(audio=reconstructed_audio, spectrogram=masked_spec)


        return separated_signals

    @classmethod
    def from_config(cls, config: BanditSeparatorConfig) -> "Bandit":
        """
        Instantiates a Bandit model from a BanditSeparatorConfig Pydantic model.

        Args:
            config (BanditSeparatorConfig): A BanditSeparatorConfig Pydantic object containing
                                       all necessary parameters for the model.

        Returns:
            Bandit: An instance of the Bandit model.
        """
        return cls(config)

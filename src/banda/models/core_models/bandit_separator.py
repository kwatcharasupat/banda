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
import hydra.utils

from banda.data.batch_types import FixedStemSeparationBatch
from banda.core.interfaces import BaseModel
from banda.models.common_components.core_components.base_separator import BaseBandsplitSeparator
from banda.models.common_components.mask_estimation.mask_estimation_modules import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.common_components.utils.constants import VDBO_STEMS # Example default stems
from banda.models.common_components.configs.common_configs import BanditSeparatorConfig, STFTConfig, BandsplitModuleConfig, TFModelConfig, MaskEstimationConfig

@MODELS_REGISTRY.register("bandit")

class Bandit(BaseBandsplitSeparator):
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

        self.stems: List[str] = config.stems
        self.fs: int = config.fs
        self.emb_dim: int = config.emb_dim

        # 2. Time-Frequency Model
        tf_model_cls = MODELS_REGISTRY.get(config.tfmodel_config.tf_model_type)
        self.tf_model = tf_model_cls.from_config(config.tfmodel_config)

        # 3. Mask Estimation Module
        # Adjust n_freq for MaskEstimationModule if DC band is dropped
        n_freq_for_mask_estim = config.stft_config.n_fft // 2 + 1
        if self.drop_dc_band:
            n_freq_for_mask_estim -= 1

        mask_estim_cls: Union[Type[MaskEstimationModule], Type[OverlappingMaskEstimationModule]] = OverlappingMaskEstimationModule if self.band_specs.is_overlapping else MaskEstimationModule
        self.mask_estim: nn.ModuleDict = nn.ModuleDict(
            {
                stem: mask_estim_cls(
                    band_specs=self.band_specs.get_band_specs(), # Pass the actual band specs
                    emb_dim=config.emb_dim,
                    mlp_dim=config.mask_estim_config.mlp_dim,
                    in_channel=config.in_channel,
                    hidden_activation=config.mask_estim_config.hidden_activation,
                    hidden_activation_kwargs=config.mask_estim_config.hidden_activation_kwargs,
                    complex_mask=config.mask_estim_config.complex_mask,
                    # Parameters specific to OverlappingMaskEstimationModule
                    **({"n_freq": n_freq_for_mask_estim} if self.band_specs.is_overlapping else {}),
                    # Removed use_freq_weights and freq_weights as they are deprecated
                )
                for stem in self.stems
            }
        )
        # 

    def forward(self, mixture_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the separator model.

        Args:
            mixture_audio (torch.Tensor): The input mixture audio tensor.
                Shape: (batch_size, channels, samples)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                                     Each tensor has shape: (batch_size, channels, samples)
        """
        input_to_bandsplit, original_audio_length, mixture_spec = self._process_spectrogram(mixture_audio)
        band_features = self._process_bandsplit(input_to_bandsplit)

        # 3. Time-Frequency Model: Process band features
        # Output shape: (batch, n_bands, n_time, emb_dim)
        processed_features: torch.Tensor = self.tf_model(band_features)
        if torch.isnan(processed_features).any():
            raise ValueError("NaN in processed_features")

        # 4. Mask Estimation: Predict masks for each source
        # Output shape: Dict[stem_name, (batch, in_channel, n_freq, n_time)]
        masks: Dict[str, torch.Tensor] = {
            stem: self.mask_estim[stem](processed_features) # Shape: (batch_size, in_channel, freq_bins, time_frames)
            for stem in self.stems
        }

        # 5. Apply Masks and ISTFT: Reconstruct separated audio
        separated_audio: Dict[str, torch.Tensor] = {}
        for stem, mask in masks.items():
            if torch.isnan(mask).any():
                raise ValueError(f"NaN in mask for stem {stem}")
            
            # Apply mask to the mixture spectrogram
            # mixture_spec is (batch, channels, freq_bins, time_frames) - complex
            # mask is (batch, in_channel, freq_bins, time_frames) - complex or real
            # Need to ensure mask is complex if mixture_spec is complex
            if mixture_spec.is_complex() and not mask.is_complex():
                # If mask is real, convert it to complex with zero imaginary part
                mask = torch.complex(mask, torch.zeros_like(mask))

            masked_spec: torch.Tensor = mixture_spec * mask # Shape: (batch_size, channels, freq_bins, time_frames)
            if torch.isnan(masked_spec).any():
                raise ValueError(f"NaN in masked_spec for stem {stem}")

            separated_audio[stem] = self._reconstruct_audio(masked_spec, original_audio_length)


        return separated_audio

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Bandit":
        """
        Instantiates a Bandit model from a DictConfig.

        This class method is responsible for parsing the configuration and
        constructing a `BanditSeparatorConfig` object, which is then used to
        initialize the `Bandit` model.

        Args:
            cfg (DictConfig): A DictConfig object containing the model configuration.

        Returns:
            Bandit: An instance of the Bandit model.
        """
        model_cfg: DictConfig = cfg # Access the model subtree


        # Extract top-level parameters from model_cfg
        stems: List[str] = model_cfg.get("stems", VDBO_STEMS)
        fs: int = model_cfg.get("fs", 44100)
        emb_dim: int = model_cfg.get("emb_dim", 128)
        in_channel: int = model_cfg.get("in_channel", 2)
        drop_dc_band: bool = model_cfg.get("drop_dc_band", True) # Get from config, default to True

        # Instantiate sub-configs by first resolving their DictConfig to a plain dict
        # and then instantiating the Pydantic model from that dict.
        stft_config: STFTConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.stft, resolve=True))
        tfmodel_config: TFModelConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.tfmodel, resolve=True))
        mask_estim_config: MaskEstimationConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.mask_estim, resolve=True))

        bandsplit_config: BandsplitModuleConfig = cls._instantiate_bandsplit_config(model_cfg, stft_config)

        # Create the main BanditSeparatorConfig instance
        separator_config: BanditSeparatorConfig = BanditSeparatorConfig(
            stems=stems,
            fs=fs,
            emb_dim=emb_dim,
            in_channel=in_channel,
            stft_config=stft_config,
            bandsplit_config=bandsplit_config,
            tfmodel_config=tfmodel_config,
            mask_estim_config=mask_estim_config,
        )

        return cls(separator_config)
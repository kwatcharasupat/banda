#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Type
from omegaconf import DictConfig, OmegaConf
from banda.utils.registry import MODELS_REGISTRY, QUERY_MODELS_REGISTRY, QUERY_PROCESSORS_REGISTRY
import hydra.utils
import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

from banda.core.interfaces import BaseQueryModel, BaseTimeFrequencyModel
from banda.models.common_components.core_components.base_separator import BaseBandsplitSeparator
from banda.models.common_components.mask_estimation.mask_estimation_modules import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.common_components.utils.constants import VDBO_STEMS # Example default stems
from banda.models.common_components.configs.common_configs import BanquetSeparatorConfig, STFTConfig, BandsplitModuleConfig, TFModelConfig, MaskEstimationConfig
from pydantic import Field
from dataclasses import dataclass


@dataclass
class QuerySeparatorConfig(BanquetSeparatorConfig): # This will be renamed to BanquetSeparatorConfig later
    """
    Configuration for the QuerySeparator model.
    Extends SeparatorConfig with query-specific parameters.
    """
    query_features: int = Field(..., description="Number of features in the query tensor.")
    query_processor_type: str = Field("linear", description="Type of query processor (e.g., 'linear', 'mlp').")


@QUERY_MODELS_REGISTRY.register("banquet")
class Banquet(BaseBandsplitSeparator, BaseQueryModel):
    """
    A query-based source separation model adapted from the Separator.

    This model takes a mixture audio and a query as input, and outputs separated sources
    using a bandsplit approach with a time-frequency model and mask estimation,
    conditioned by the query.
    It inherits from BaseQueryModel for general query model interface and SpectralComponent
    for STFT/ISTFT functionalities.
    """

    def __init__(self, config: BanquetSeparatorConfig) -> None:
        """
        Initializes the QuerySeparator model.

        Args:
            config (BanquetSeparatorConfig): A Pydantic configuration object containing
                                          all necessary parameters for the model.

        Raises:
            ValueError: If `band_specs` is not provided in the configuration.
            ValueError: If an unknown `tf_model_type` is specified in the configuration.
            ValueError: If an unknown `query_processor_type` is specified.
        """
        super().__init__(config) # Initialize BaseBandsplitSeparator first
        BaseQueryModel.__init__(self) # Initialize BaseQueryModel


        self.stems: List[str] = config.stems
        self.query_features: int = config.query_features
        self.emb_dim: int = config.emb_dim

        # 2. Time-Frequency Model
        # Instantiate the TF model here, after config validation
        self.tf_model: BaseTimeFrequencyModel = hydra.utils.instantiate(config.tf_model)

        # 3. Query Processor
        query_processor_cls = QUERY_PROCESSORS_REGISTRY.get(config.query_processor_type)
        self.query_processor = query_processor_cls(
            query_features=self.query_features,
            emb_dim=self.emb_dim
        )

        # 4. Mask Estimation Module
        # Adjust n_freq for MaskEstimationModule if DC band is dropped
        n_freq_for_mask_estim = config.stft_config.n_fft // 2 + 1
        if self.drop_dc_band:
            n_freq_for_mask_estim -= 1

        mask_estim_cls: Union[Type[MaskEstimationModule], Type[OverlappingMaskEstimationModule]] = OverlappingMaskEstimationModule if self.band_specs.is_overlapping else MaskEstimationModule
        # Instantiate mask_estim for a single stem
        self.mask_estim = mask_estim_cls(
            band_specs=self.band_specs.get_band_specs(), # Pass the actual band specs
            emb_dim=config.emb_dim,
            mlp_dim=config.mask_estim_config.mlp_dim,
            in_channel=config.in_channel, # This is correct for a single stem
            hidden_activation=config.mask_estim_config.hidden_activation,
            hidden_activation_kwargs=config.mask_estim_config.hidden_activation_kwargs,
            complex_mask=config.mask_estim_config.complex_mask,
            # Parameters specific to OverlappingMaskEstimationModule
            **({"n_freq": n_freq_for_mask_estim} if self.band_specs.is_overlapping else {}),
            # Removed use_freq_weights and freq_weights as they are deprecated
        )

    def forward(self, mixture_audio: torch.Tensor, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the QuerySeparator model for source separation.

        Args:
            mixture_audio (torch.Tensor): The input mixture audio tensor.
                Shape: (batch_size, channels, samples)
            query (torch.Tensor): Query tensor for conditioning the separation.
                Shape: (batch_size, query_features)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                                     Each tensor has shape: (batch_size, channels, samples)
        """
        # Call the internal forward implementation from BaseBandsplitSeparator
        # Corrected: Use _process_spectrogram which returns 3 values
        input_to_bandsplit, original_audio_length, raw_mixture_spec = self._process_spectrogram(mixture_audio)
        
        band_features = self._process_bandsplit(input_to_bandsplit)

        # Process the query and integrate it with band features
        processed_query: torch.Tensor = self.query_processor(query)
        processed_query_expanded = processed_query.unsqueeze(1).unsqueeze(1)
        conditioned_features: torch.Tensor = band_features + processed_query_expanded


        # Time-Frequency Model: Process conditioned band features
        processed_features: torch.Tensor = self.tf_model(conditioned_features)

        # Mask Estimation: Predict masks for each source
        masks: Dict[str, torch.Tensor] = {}
        for stem in self.stems:
            mask = self.mask_estim(processed_features)
            masks[stem] = mask


        # Apply Masks and ISTFT: Reconstruct separated audio
        separated_audio: Dict[str, torch.Tensor] = {}
        for stem, mask in masks.items():
            # Convert raw_mixture_spec to complex before multiplication
            raw_mixture_spec_complex = torch.view_as_complex(raw_mixture_spec)
            
            # mask is (batch, channels, freq, time)
            # raw_mixture_spec_complex is (batch, channels, freq, time)
            # No unsqueeze needed for mask as it already matches the dimensions for element-wise multiplication
            masked_spec_complex: torch.Tensor = raw_mixture_spec_complex * mask
            
            # Convert back to real tensor for _reconstruct_audio
            masked_spec: torch.Tensor = torch.view_as_real(masked_spec_complex)
            
            separated_audio[stem] = self._reconstruct_audio(masked_spec, original_audio_length)

        return separated_audio

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Banquet":
        """
        Instantiates a BanquetSeparator model from a DictConfig.

        This class method is responsible for parsing the configuration and
        constructing a `BanquetSeparatorConfig` object, which is then used to
        initialize the `Banquet` model.

        Args:
            cfg (DictConfig): A DictConfig object containing the model configuration.

        Returns:
            Banquet: An instance of the Banquet model.
        """
        model_cfg: DictConfig = cfg # Access the model subtree


        # Extract top-level parameters from model_cfg
        stems: List[str] = model_cfg.get("stems", VDBO_STEMS)
        fs: int = model_cfg.get("fs", 44100)
        emb_dim: int = model_cfg.get("emb_dim", 128)
        in_channel: int = model_cfg.get("in_channel", 2)
        query_features: int = model_cfg.get("query_features")
        query_processor_type: str = model_cfg.get("query_processor_type", "linear")


        # Instantiate sub-configs by first resolving their DictConfig to a plain dict.
        # Note: tfmodel is now directly instantiated as a module, not a config.
        stft_config: STFTConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.stft, resolve=True))
        mask_estim_config: MaskEstimationConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.mask_estim, resolve=True))

        bandsplit_config: BandsplitModuleConfig = cls._instantiate_bandsplit_config(model_cfg, stft_config)

        # Create the main BanquetSeparatorConfig instance
        banquet_separator_config: BanquetSeparatorConfig = BanquetSeparatorConfig(
            stems=stems,
            fs=fs,
            emb_dim=emb_dim,
            in_channel=in_channel,
            stft_config=stft_config,
            bandsplit_config=bandsplit_config,
            tf_model=model_cfg.tfmodel, # Pass the DictConfig directly
            mask_estim_config=mask_estim_config,
            query_features=query_features,
            query_processor_type=query_processor_type,
        )

        return cls(banquet_separator_config)
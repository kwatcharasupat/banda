#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#  For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
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
from banda.models.core_models.base_separator import BaseSeparator
from banda.models.common_components.mask_estimation.mask_estimation_modules import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.common_components.utils.constants import VDBO_STEMS # Example default stems
from banda.models.common_components.configs.common_configs import BanquetSeparatorConfig, STFTConfig, BandsplitModuleConfig, BaseTFModelConfig
from pydantic import Field
from dataclasses import dataclass
from banda.data.batch_types import AudioSignal, QuerySignal
from banda.models.common_components.spectral_components.spectral_base import BandsplitSpecification


@QUERY_MODELS_REGISTRY.register("banquet")
class Banquet(BaseSeparator, BaseQueryModel):
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
        super().__init__(config) # Initialize BaseSeparator and BaseQueryModel via super()

        # Query Processor
        query_processor_cls = QUERY_PROCESSORS_REGISTRY.get(config.query_processor_type)
        self.query_processor = query_processor_cls(
            query_features=config.query_features,
            emb_dim=self.emb_dim
        )

        # Mask Estimation Module
        # This part is already handled in BaseSeparator's __init__
        # self.mask_estim = self.mask_estim_cls.from_config(
        #     OmegaConf.create(self.mask_estim_config_params)
        # )

    def forward(self, mixture_audio: torch.Tensor, query: QuerySignal) -> Dict[str, AudioSignal]:
        """
        Forward pass of the QuerySeparator model for source separation.

        Args:
            mixture_audio (torch.Tensor): The input mixture audio tensor.
                Shape: (batch_size, channels, samples)
            query (QuerySignal): Query signal for conditioning the separation.

        Returns:
            Dict[str, AudioSignal]: Dictionary of separated source AudioSignal objects.
        """
        # Process the mixture audio to get its spectrogram representation
        mixture_signal: AudioSignal = self._apply_stft(mixture_audio)
        
        # Extract the spectrogram from the mixture_signal for bandsplit processing
        input_to_bandsplit = mixture_signal.spectrogram

        band_features = self._apply_bandsplit(input_to_bandsplit) # Changed from _process_bandsplit

        # Process the query using the query processor
        processed_query: torch.Tensor = self.query_processor(query)

        processed_query_expanded = processed_query.unsqueeze(1).unsqueeze(1)
        conditioned_features: torch.Tensor = band_features + processed_query_expanded


        # Time-Frequency Model: Process conditioned band features
        processed_features: torch.Tensor = self._apply_tf_model(conditioned_features) # Changed from self.tf_model

        # Mask Estimation: Predict masks for each source
        masks: Dict[str, torch.Tensor] = {}
        for stem in self.stems:
            mask = self._apply_mask_estimation(processed_features) # Changed from self.mask_estim
            masks[stem] = mask


        # Apply Masks and ISTFT: Reconstruct separated audio
        separated_signals: Dict[str, AudioSignal] = {}
        for stem, mask in masks.items():
            # Use the original mixture spectrogram for masking
            raw_mixture_spec = mixture_signal.spectrogram
            
            # Convert raw_mixture_spec to complex before multiplication
            raw_mixture_spec_complex = torch.view_as_complex(raw_mixture_spec)
            
            # mask is (batch, channels, freq, time)
            # raw_mixture_spec_complex is (batch, channels, freq, time)
            # No unsqueeze needed for mask as it already matches the dimensions for element-wise multiplication
            masked_spec_complex: torch.Tensor = raw_mixture_spec_complex * mask
            
            # Convert back to real tensor for _reconstruct_audio
            masked_spec: torch.Tensor = torch.view_as_real(masked_spec_complex)
            
            # Reconstruct audio from the masked spectrogram
            reconstructed_audio = self._apply_istft(masked_spec, mixture_audio.shape[-1]) # Changed from _reconstruct_audio_from_masked_spec
            
            # Create an AudioSignal object for the separated source
            separated_signals[stem] = AudioSignal(audio=reconstructed_audio, spectrogram=masked_spec)

        return separated_signals

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Banquet":
        """
        Instantiates a BanquetSeparator model from a DictConfig.

        This class method is responsible for parsing the configuration and
        constructing a `BanquetSeparatorConfig` object, which is then used to
        initialize the `Banquet` model.

        Args:
            cfg (DictConfig): A DictConfig object containing the full configuration.

        Returns:
            Banquet: An instance of the Banquet model.
        """
        # Extract relevant parts of the config
        model_cfg = cfg.model
        data_cfg = cfg.data

        # Instantiate bandsplit module config, allowing recursive instantiation for band_specs
        # Removed _recursive_=False to allow Hydra to fully instantiate nested Pydantic models
        bandsplit_module_config = hydra.utils.instantiate(model_cfg.bandsplit)

        # Manually construct a dictionary that matches BanquetSeparatorConfig's structure
        # and instantiate nested Pydantic models using hydra.utils.instantiate
        banquet_config_data = {
            "stems": VDBO_STEMS, # Default stems for Banquet
            "fs": data_cfg.train_dataset_config.config.fs,
            "emb_dim": model_cfg.tfmodel.emb_dim,
            "in_channel": model_cfg.input_channels,
            "query_features": data_cfg.query_features, # From data config
            "query_processor_type": model_cfg.query_processor_type if "query_processor_type" in model_cfg else "linear", # Default to linear if not specified
            "stft": hydra.utils.instantiate(model_cfg.stft, _recursive_=False), # Instantiate STFTConfig
            "bandsplit": bandsplit_module_config, # Use the already instantiated bandsplit config
            "tfmodel": hydra.utils.instantiate(model_cfg.tfmodel, _recursive_=False), # Instantiate TFModelConfig (RNN, Transformer, or Mamba)
            "mask_estim": hydra.utils.instantiate(model_cfg.mask_estim, _recursive_=False), # Instantiate MaskEstimationConfig
        }
        
        # Create the BanquetSeparatorConfig instance
        banquet_separator_config: BanquetSeparatorConfig = BanquetSeparatorConfig(**banquet_config_data)
        
        return cls(banquet_separator_config)
#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Type
from omegaconf import DictConfig, OmegaConf
from banda.utils.registry import MODELS_REGISTRY, QUERY_MODELS_REGISTRY
import importlib
import hydra.utils

from banda.data.batch_types import FixedStemSeparationBatch
from banda.core.interfaces import BaseModel, BaseQueryModel
from banda.models.spectral import SpectralComponent
from banda.models.components.bandsplit import BandSplitModule
from banda.models.components.mask_estimation import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.components.time_frequency_model import SeqBandModellingModule, TransformerTimeFreqModule, ConvolutionalTimeFreqModule
from banda.models.utils import VDBO_STEMS # Example default stems
from banda.models.spectral import BandsplitSpecification # Import BandsplitSpecification (now in models folder)
from banda.models.spectral import get_bandsplit_specs_factory
from banda.models.configs import SeparatorConfig, STFTConfig, BandsplitModuleConfig, TFModelConfig, MaskEstimationConfig
from pydantic import Field
from dataclasses import dataclass


@dataclass
class QuerySeparatorConfig(SeparatorConfig):
    """
    Configuration for the QuerySeparator model.
    Extends SeparatorConfig with query-specific parameters.
    """
    query_features: int = Field(..., description="Number of features in the query tensor.")
    query_processor_type: str = Field("linear", description="Type of query processor (e.g., 'linear', 'mlp').")


@QUERY_MODELS_REGISTRY.register("query_separator")
class QuerySeparator(BaseQueryModel, SpectralComponent):
    """
    A query-based source separation model adapted from the Separator.

    This model takes a mixture audio and a query as input, and outputs separated sources
    using a bandsplit approach with a time-frequency model and mask estimation,
    conditioned by the query.
    It inherits from BaseQueryModel for general query model interface and SpectralComponent
    for STFT/ISTFT functionalities.
    """

    def __init__(self, config: QuerySeparatorConfig) -> None:
        """
        Initializes the QuerySeparator model.

        Args:
            config (QuerySeparatorConfig): A Pydantic configuration object containing
                                          all necessary parameters for the model.

        Raises:
            ValueError: If `band_specs` is not provided in the configuration.
            ValueError: If an unknown `tf_model_type` is specified in the configuration.
            ValueError: If an unknown `query_processor_type` is specified.
        """
        BaseQueryModel.__init__(self) # Initialize BaseQueryModel first
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

        self.stems: List[str] = config.stems
        self.band_specs: BandsplitSpecification = config.bandsplit_config.band_specs
        self.fs: int = config.fs
        self.n_fft: int = config.stft_config.n_fft # Use n_fft from STFT config
        self.in_channel: int = config.in_channel
        self.query_features: int = config.query_features
        self.emb_dim: int = config.emb_dim

        # 1. Bandsplit Module
        if self.band_specs is None:
            raise ValueError("band_specs must be provided to Separator.")

        self.band_split: BandSplitModule = BandSplitModule(
            band_spec_obj=self.band_specs, # Pass the BandsplitSpecification object
            emb_dim=config.emb_dim,
            in_channel=config.in_channel,
            require_no_overlap=config.bandsplit_config.require_no_overlap,
            require_no_gap=config.bandsplit_config.require_no_gap,
            normalize_channel_independently=config.bandsplit_config.normalize_channel_independently,
            treat_channel_as_feature=config.bandsplit_config.treat_channel_as_feature,
        )

        # 2. Time-Frequency Model
        if config.tfmodel_config.tf_model_type == "rnn":
            self.tf_model: SeqBandModellingModule = SeqBandModellingModule(
                n_modules=config.tfmodel_config.n_tf_modules,
                emb_dim=config.emb_dim,
                rnn_dim=config.tfmodel_config.rnn_dim,
                bidirectional=config.tfmodel_config.bidirectional,
                rnn_type=config.tfmodel_config.rnn_type,
            )
        elif config.tfmodel_config.tf_model_type == "transformer":
            self.tf_model: TransformerTimeFreqModule = TransformerTimeFreqModule(
                n_modules=config.tfmodel_config.n_tf_modules,
                emb_dim=config.emb_dim,
                rnn_dim=config.tfmodel_config.rnn_dim, # Not used, but for compatibility
                bidirectional=config.tfmodel_config.bidirectional, # Not used, but for compatibility
                dropout=config.tfmodel_config.tf_dropout,
            )
        elif config.tfmodel_config.tf_model_type == "conv":
            self.tf_model: ConvolutionalTimeFreqModule = ConvolutionalTimeFreqModule(
                n_modules=config.tfmodel_config.n_tf_modules,
                emb_dim=config.emb_dim,
                rnn_dim=config.tfmodel_config.rnn_dim, # Not used, but for compatibility
                bidirectional=config.tfmodel_config.bidirectional, # Not used, but for compatibility
                dropout=config.tfmodel_config.tf_dropout,
            )
        else:
            raise ValueError(f"Unknown tf_model_type: {config.tfmodel_config.tf_model_type}")

        # 3. Query Processor
        if config.query_processor_type == "linear":
            self.query_processor = nn.Linear(self.query_features, self.emb_dim)
        elif config.query_processor_type == "mlp":
            # Example MLP, can be expanded
            self.query_processor = nn.Sequential(
                nn.Linear(self.query_features, self.emb_dim * 2),
                nn.ReLU(),
                nn.Linear(self.emb_dim * 2, self.emb_dim)
            )
        else:
            raise ValueError(f"Unknown query_processor_type: {config.query_processor_type}")


        # 4. Mask Estimation Module
        print(f"QuerySeparator: is_overlapping = {self.band_specs.is_overlapping}")
        print(f"QuerySeparator: n_fft // 2 + 1 = {config.stft_config.n_fft // 2 + 1}")
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
                    **({"n_freq": config.stft_config.n_fft // 2 + 1} if self.band_specs.is_overlapping else {}),
                    # Removed use_freq_weights and freq_weights as they are deprecated
                )
                for stem in self.stems
            }
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
        # Store original audio length for ISTFT
        original_audio_length: int = mixture_audio.shape[-1]

        # 1. STFT: Convert audio to complex spectrogram
        # Output shape: (batch_size, channels, freq_bins, time_frames, 2)
        mixture_spec: torch.Tensor = self.stft(mixture_audio) # Shape: (batch_size, channels, freq_bins, time_frames, 2)


        # Convert complex spectrogram to real-view for bandsplit input
        # (batch, channels, freq_bins, time_frames, 2) -> (batch, channels * 2, freq_bins, time_frames)
        input_to_bandsplit: torch.Tensor = torch.cat([mixture_spec.real, mixture_spec.imag], dim=1) # Shape: (batch_size, channels * 2, freq_bins, time_frames)


        # 2. Bandsplit: Process each frequency band
        # Output shape: (batch, n_bands, n_time, emb_dim)
        band_features: torch.Tensor = self.band_split(input_to_bandsplit) # Shape: (batch_size, n_bands, time_frames, emb_dim)

        # Process the query and integrate it with band features
        # query: (batch_size, query_features) -> processed_query: (batch_size, emb_dim)
        processed_query: torch.Tensor = self.query_processor(query)

        # Expand processed_query to match the dimensions of band_features for broadcasting
        # processed_query: (batch_size, emb_dim) -> (batch_size, 1, 1, emb_dim)
        processed_query_expanded = processed_query.unsqueeze(1).unsqueeze(1)

        # Simple integration: add processed query to band features.
        # More complex integrations (e.g., attention) would be used in a real model.
        conditioned_features: torch.Tensor = band_features + processed_query_expanded # Shape: (batch_size, n_bands, time_frames, emb_dim)


        # 3. Time-Frequency Model: Process conditioned band features
        # Output shape: (batch, n_bands, n_time, emb_dim)
        processed_features: torch.Tensor = self.tf_model(conditioned_features) # Shape: (batch_size, n_bands, time_frames, emb_dim)

        # 4. Mask Estimation: Predict masks for each source
        # Output shape: Dict[stem_name, (batch, in_channel, n_freq, n_time)]
        masks: Dict[str, torch.Tensor] = {
            stem: self.mask_estim[stem](processed_features) # Shape: (batch_size, in_channel, freq_bins, time_frames)
            for stem in self.stems
        }

        # 5. Apply Masks and ISTFT: Reconstruct separated audio
        separated_audio: Dict[str, torch.Tensor] = {}
        for stem, mask in masks.items():
            # Apply mask to the mixture spectrogram
            # Ensure mask and mixture_spec are compatible for element-wise multiplication
            # (batch, in_channel, n_freq, n_time, 2) * (batch, in_channel, n_freq, n_time)
            # Need to expand mask to have the real/imaginary dimension=
            masked_spec: torch.Tensor = mixture_spec * mask.unsqueeze(-1) # Shape: (batch_size, channels, freq_bins, time_frames, 2)

            # ISTFT: Convert masked spectrogram back to audio
            separated_audio[stem] = self.inverse(masked_spec, original_audio_length) # Shape: (batch_size, channels, samples)

        return separated_audio

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "QuerySeparator":
        """
        Instantiates a QuerySeparator model from a DictConfig.

        This class method is responsible for parsing the configuration and
        constructing a `QuerySeparatorConfig` object, which is then used to
        initialize the `QuerySeparator` model.

        Args:
            cfg (DictConfig): A DictConfig object containing the model configuration.

        Returns:
            QuerySeparator: An instance of the QuerySeparator model.
        """
        model_cfg: DictConfig = cfg # Access the model subtree


        # Extract top-level parameters from model_cfg
        stems: List[str] = model_cfg.get("stems", VDBO_STEMS)
        fs: int = model_cfg.get("fs", 44100)
        emb_dim: int = model_cfg.get("emb_dim", 128)
        in_channel: int = model_cfg.get("in_channel", 2)
        query_features: int = model_cfg.get("query_features")
        query_processor_type: str = model_cfg.get("query_processor_type", "linear")


        # Instantiate sub-configs by first resolving their DictConfig to a plain dict
        # and then instantiating the Pydantic model from that dict.
        stft_config: STFTConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.stft, resolve=True))
        tfmodel_config: TFModelConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.tfmodel, resolve=True))
        mask_estim_config: MaskEstimationConfig = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.mask_estim, resolve=True))

        # Instantiate band_specs first, as it requires n_fft and fs
        bandsplit_type = model_cfg.bandsplit.band_specs.bandsplit_type
        n_bands = model_cfg.bandsplit.band_specs.n_bands
        band_specs: BandsplitSpecification = get_bandsplit_specs_factory(
            bandsplit_type=bandsplit_type,
            n_fft=stft_config.n_fft,
            fs=fs,
            n_bands=n_bands
        )

        # Instantiate bandsplit_config using hydra.utils.instantiate
        # Manually create bandsplit_config, passing the instantiated band_specs
        bandsplit_config: BandsplitModuleConfig = hydra.utils.instantiate(
            model_cfg.bandsplit,
            band_specs=band_specs, # Pass the already instantiated band_specs
            _recursive_=False # Prevent re-instantiation of band_specs
        )

        # Create the main QuerySeparatorConfig instance
        query_separator_config: QuerySeparatorConfig = QuerySeparatorConfig(
            stems=stems,
            fs=fs,
            emb_dim=emb_dim,
            in_channel=in_channel,
            stft_config=stft_config,
            bandsplit_config=bandsplit_config,
            tfmodel_config=tfmodel_config,
            mask_estim_config=mask_estim_config,
            query_features=query_features,
            query_processor_type=query_processor_type,
        )

        return cls(query_separator_config)
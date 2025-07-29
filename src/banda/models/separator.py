#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#


import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import importlib
import hydra.utils # Added this import

from banda.data.batch_types import SeparationBatch, FixedStemSeparationBatch
from banda.models.spectral import SpectralComponent
from banda.models.components.bandsplit import BandSplitModule
from banda.models.components.mask_estimation import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.components.time_frequency_model import SeqBandModellingModule, TransformerTimeFreqModule, ConvolutionalTimeFreqModule
from banda.models.utils import VDBO_STEMS # Example default stems
from banda.models.spectral import BandsplitSpecification # Import BandsplitSpecification (now in models folder)
from banda.models.configs import SeparatorConfig, STFTConfig, BandsplitModuleConfig, TFModelConfig, MaskEstimationConfig


class Separator(SpectralComponent):
    """
    A fixed-stem Bandit-like source separation model.

    This model takes a mixture audio as input and outputs separated sources
    using a bandsplit approach with a time-frequency model and mask estimation.
    """

    def __init__(self, config: SeparatorConfig) -> None:
        """
        Initializes the Separator model.
        """
        super().__init__(
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

        self.stems = config.stems
        self.band_specs = config.bandsplit_config.band_specs
        self.fs = config.fs
        self.n_fft = config.stft_config.n_fft # Use n_fft from STFT config
        self.in_channel = config.in_channel

        # 1. Bandsplit Module
        if self.band_specs is None:
            raise ValueError("band_specs must be provided to Separator.")

        self.band_split = BandSplitModule(
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
            self.tf_model = SeqBandModellingModule(
                n_modules=config.tfmodel_config.n_tf_modules,
                emb_dim=config.emb_dim,
                rnn_dim=config.tfmodel_config.rnn_dim,
                bidirectional=config.tfmodel_config.bidirectional,
                rnn_type=config.tfmodel_config.rnn_type,
            )
        elif config.tfmodel_config.tf_model_type == "transformer":
            self.tf_model = TransformerTimeFreqModule(
                n_modules=config.tfmodel_config.n_tf_modules,
                emb_dim=config.emb_dim,
                rnn_dim=config.tfmodel_config.rnn_dim, # Not used, but for compatibility
                bidirectional=config.tfmodel_config.bidirectional, # Not used, but for compatibility
                dropout=config.tfmodel_config.tf_dropout,
            )
        elif config.tfmodel_config.tf_model_type == "conv":
            self.tf_model = ConvolutionalTimeFreqModule(
                n_modules=config.tfmodel_config.n_tf_modules,
                emb_dim=config.emb_dim,
                rnn_dim=config.tfmodel_config.rnn_dim, # Not used, but for compatibility
                bidirectional=config.tfmodel_config.bidirectional, # Not used, but for compatibility
                dropout=config.tfmodel_config.tf_dropout,
            )
        else:
            raise ValueError(f"Unknown tf_model_type: {config.tfmodel_config.tf_model_type}")

        # 3. Mask Estimation Module
        mask_estim_cls = OverlappingMaskEstimationModule if self.band_specs.is_overlapping else MaskEstimationModule
        self.mask_estim = nn.ModuleDict(
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

    def forward(self, mixture_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the separator model.

        Args:
            mixture_audio (torch.Tensor): The input mixture audio tensor.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                                     Each tensor has shape: (batch_size, channels, samples)
        """
        # Store original audio length for ISTFT
        original_audio_length = mixture_audio.shape[-1]

        # 1. STFT: Convert audio to complex spectrogram
        # Output shape: (batch_size, channels, freq_bins, time_frames, 2)
        mixture_spec = self.stft(mixture_audio)
        
        

        # Convert complex spectrogram to real-view for encoder (batch, channels, freq, time, 2) -> (batch, channels, freq, time)
        # Assuming the encoder expects a real-valued input, e.g., magnitude spectrogram or concatenated real/imag
        # Let's concatenate real and imaginary parts along the channel dimension
        # (batch, channels, freq_bins, time_frames, 2) -> (batch, channels * 2, freq_bins, time_frames)
        
        input_to_bandsplit = torch.cat([mixture_spec.real, mixture_spec.imag], dim=1)


        # 2. Bandsplit: Process each frequency band
        # Output shape: (batch, n_bands, n_time, emb_dim)
        band_features = self.band_split(input_to_bandsplit)

        # 3. Time-Frequency Model: Process band features
        # Output shape: (batch, n_bands, n_time, emb_dim)
        processed_features = self.tf_model(band_features)

        # 4. Mask Estimation: Predict masks for each source
        # Output shape: Dict[stem_name, (batch, in_channel, n_freq, n_time)]
        masks = {
            stem: self.mask_estim[stem](processed_features)
            for stem in self.stems
        }

        # 5. Apply Masks and ISTFT: Reconstruct separated audio
        separated_audio = {}
        for stem, mask in masks.items():
            # Apply mask to the mixture spectrogram
            # Ensure mask and mixture_spec are compatible for element-wise multiplication
            # (batch, in_channel, n_freq, n_time, 2) * (batch, in_channel, n_freq, n_time)
            # Need to expand mask to have the real/imaginary dimension=
            masked_spec = mixture_spec * mask # Add a dimension for real/imaginary

            # ISTFT: Convert masked spectrogram back to audio
            separated_audio[stem] = self.inverse(masked_spec, original_audio_length)

        return separated_audio

    @classmethod
    def from_config(cls, cfg: DictConfig): # Changed config to cfg
        """
        Instantiates a Separator model from a DictConfig.
        """
        model_cfg = cfg # Access the model subtree

        
        # Extract top-level parameters from model_cfg
        stems = model_cfg.get("stems", VDBO_STEMS)
        fs = model_cfg.get("fs", 44100)
        emb_dim = model_cfg.get("emb_dim", 128)
        in_channel = model_cfg.get("in_channel", 2)

        # Instantiate sub-configs by first resolving their DictConfig to a plain dict
        # and then instantiating the Pydantic model from that dict.
        stft_config = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.stft, resolve=True))
        tfmodel_config = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.tfmodel, resolve=True))
        mask_estim_config = hydra.utils.instantiate(OmegaConf.to_container(model_cfg.mask_estim, resolve=True))

        # Instantiate band_specs first, as it requires n_fft and fs
        band_specs = hydra.utils.instantiate(
            model_cfg.bandsplit.band_specs,
            n_fft=stft_config.n_fft,
            fs=fs
        )

        # Manually create bandsplit_config, passing the instantiated band_specs
        bandsplit_config = BandsplitModuleConfig(
            band_specs=band_specs,
            require_no_overlap=model_cfg.bandsplit.require_no_overlap,
            require_no_gap=model_cfg.bandsplit.require_no_gap,
            normalize_channel_independently=model_cfg.bandsplit.normalize_channel_independently,
            treat_channel_as_feature=model_cfg.bandsplit.treat_channel_as_feature,
        )

        # Create the main SeparatorConfig instance
        separator_config = SeparatorConfig(
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
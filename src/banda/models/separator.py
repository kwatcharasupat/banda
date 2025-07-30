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
import importlib
import hydra.utils # Added this import
import structlog

logger = structlog.get_logger(__name__)

from banda.data.batch_types import FixedStemSeparationBatch
from banda.core.interfaces import BaseModel
from banda.models.spectral import SpectralComponent
from banda.models.components.bandsplit import BandSplitModule
from banda.models.components.mask_estimation import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.components.time_frequency_model import SeqBandModellingModule, TransformerTimeFreqModule, ConvolutionalTimeFreqModule
from banda.models.utils import VDBO_STEMS # Example default stems
from banda.models.spectral import BandsplitSpecification # Import BandsplitSpecification (now in models folder)
from banda.models.spectral import get_bandsplit_specs_factory
from banda.models.configs import SeparatorConfig, STFTConfig, BandsplitModuleConfig, TFModelConfig, MaskEstimationConfig

@MODELS_REGISTRY.register("separator")

class Separator(BaseModel, SpectralComponent):
    """
    A fixed-stem Bandit-like source separation model.

    This model takes a mixture audio as input and outputs separated sources
    using a bandsplit approach with a time-frequency model and mask estimation.
    It inherits from BaseModel for general model interface and SpectralComponent
    for STFT/ISTFT functionalities.
    """

    def __init__(self, config: SeparatorConfig) -> None:
        """
        Initializes the Separator model.

        Args:
            config (SeparatorConfig): A Pydantic configuration object containing
                                      all necessary parameters for the model.

        Raises:
            ValueError: If `band_specs` is not provided in the configuration.
            ValueError: If an unknown `tf_model_type` is specified in the configuration.
        """
        BaseModel.__init__(self) # Initialize BaseModel first
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
        self.drop_dc_band: bool = True # Hardcode for now, will be configurable later

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

        # 3. Mask Estimation Module
        print(f"Separator: is_overlapping = {self.band_specs.is_overlapping}")
        # Adjust n_freq for MaskEstimationModule if DC band is dropped
        n_freq_for_mask_estim = config.stft_config.n_fft // 2 + 1
        if self.drop_dc_band:
            n_freq_for_mask_estim -= 1
        print(f"Separator: n_freq_for_mask_estim = {n_freq_for_mask_estim}")

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
        # Store original audio length for ISTFT
        original_audio_length: int = mixture_audio.shape[-1]

        # Global Z-normalization before STFT
        # Calculate mean and std across the entire audio tensor (batch, channels, samples)
        self.audio_mean = mixture_audio.mean()
        self.audio_std = mixture_audio.std()
        
        # Add a small epsilon to std to prevent division by zero
        normalized_mixture_audio = (mixture_audio - self.audio_mean) / (self.audio_std + 1e-8)
        
        logger.debug("Separator: Global Z-normalization applied", mean=self.audio_mean.item(), std=self.audio_std.item())
        if torch.isnan(normalized_mixture_audio).any():
            logger.error("Separator: NaN detected in normalized_mixture_audio after Z-normalization", mean_val=normalized_mixture_audio.mean().item())
            raise ValueError("NaN in normalized_mixture_audio")

        # 1. STFT: Convert audio to complex spectrogram
        # Output shape: (batch_size, channels, freq_bins, time_frames) - complex tensor
        mixture_spec: torch.Tensor = self.stft(normalized_mixture_audio) # Use normalized audio
        logger.debug(f"Separator: mixture_spec shape after STFT: {mixture_spec.shape}")
        if torch.isnan(mixture_spec).any():
            logger.error("Separator: NaN detected in mixture_spec after STFT", mean_val=mixture_spec.mean().item())
            raise ValueError("NaN in mixture_spec")

        # Drop the DC band (frequency bin 0) if configured
        if self.drop_dc_band:
            mixture_spec = mixture_spec[:, :, 1:, :] # Exclude the first frequency bin
            logger.debug(f"Separator: mixture_spec shape after dropping DC band: {mixture_spec.shape}")

        # Convert complex spectrogram to real-view for bandsplit input
        # (batch, channels, freq_bins, time_frames) -> (batch, channels * 2, freq_bins, time_frames)
        input_to_bandsplit: torch.Tensor = torch.cat([mixture_spec.real, mixture_spec.imag], dim=1) # Shape: (batch_size, channels * 2, freq_bins, time_frames)
        logger.debug(f"Separator: input_to_bandsplit shape after concatenating real/imag: {input_to_bandsplit.shape}")


        # 2. Bandsplit: Process each frequency band
        # Output shape: (batch, n_bands, n_time, emb_dim)
        band_features: torch.Tensor = self.band_split(input_to_bandsplit)
        logger.debug(f"Separator: band_features shape after BandSplit: {band_features.shape}")
        if torch.isnan(band_features).any():
            logger.error("Separator: NaN detected in band_features after BandSplit", mean_val=band_features.mean().item())
            raise ValueError("NaN in band_features")

        # 3. Time-Frequency Model: Process band features
        # Output shape: (batch, n_bands, n_time, emb_dim)
        processed_features: torch.Tensor = self.tf_model(band_features)
        logger.debug(f"Separator: processed_features shape after tf_model: {processed_features.shape}")
        if torch.isnan(processed_features).any():
            logger.error("Separator: NaN detected in processed_features after tf_model", mean_val=processed_features.mean().item())
            raise ValueError("NaN in processed_features")

        # 4. Mask Estimation: Predict masks for each source
        # Output shape: Dict[stem_name, (batch, in_channel, n_freq, n_time)]
        masks: Dict[str, torch.Tensor] = {
            stem: self.mask_estim[stem](processed_features) # Shape: (batch_size, in_channel, freq_bins, time_frames)
            for stem in self.stems
        }
        logger.debug(f"Separator: masks shape (example for first stem): {list(masks.values())[0].shape}")

        # 5. Apply Masks and ISTFT: Reconstruct separated audio
        separated_audio: Dict[str, torch.Tensor] = {}
        for stem, mask in masks.items():
            if torch.isnan(mask).any():
                logger.error(f"Separator: NaN detected in mask for stem {stem}", mean_val=mask.mean().item())
                raise ValueError(f"NaN in mask for stem {stem}")
            
            # Apply mask to the mixture spectrogram
            # mixture_spec is (batch, channels, freq_bins, time_frames) - complex
            # mask is (batch, in_channel, freq_bins, time_frames) - complex or real
            # Need to ensure mask is complex if mixture_spec is complex
            if mixture_spec.is_complex() and not mask.is_complex():
                # If mask is real, convert it to complex with zero imaginary part
                mask = torch.complex(mask, torch.zeros_like(mask))

            masked_spec: torch.Tensor = mixture_spec * mask # Shape: (batch_size, channels, freq_bins, time_frames)
            logger.debug(f"Separator: masked_spec shape for stem {stem} after masking: {masked_spec.shape}")
            if torch.isnan(masked_spec).any():
                logger.error(f"Separator: NaN detected in masked_spec for stem {stem}", mean_val=masked_spec.mean().item())
                raise ValueError(f"NaN in masked_spec for stem {stem}")

            # Re-add the DC band (as zeros) before ISTFT if it was dropped
            if self.drop_dc_band:
                batch_size, num_channels, _, time_frames = masked_spec.shape
                dc_band_zeros = torch.zeros(batch_size, num_channels, 1, time_frames,
                                            device=masked_spec.device, dtype=masked_spec.dtype)
                masked_spec = torch.cat([dc_band_zeros, masked_spec], dim=2)
                logger.debug(f"Separator: masked_spec shape after re-adding DC band: {masked_spec.shape}")

            # ISTFT: Convert masked spectrogram back to audio
            reconstructed_audio = self.inverse(masked_spec, original_audio_length)
            logger.debug(f"Separator: reconstructed_audio shape for stem {stem} after ISTFT: {reconstructed_audio.shape}")
            if torch.isnan(reconstructed_audio).any():
                logger.error(f"Separator: NaN detected in reconstructed_audio for stem {stem} after ISTFT", mean_val=reconstructed_audio.mean().item())
                raise ValueError(f"NaN in reconstructed_audio for stem {stem}")
            
            # Denormalize the separated audio
            separated_audio[stem] = (reconstructed_audio * self.audio_std) + self.audio_mean
            logger.debug(f"Separator: separated_audio shape for stem {stem} after denormalization: {separated_audio[stem].shape}")
            if torch.isnan(separated_audio[stem]).any():
                logger.error(f"Separator: NaN detected in denormalized separated_audio for stem {stem}", mean_val=separated_audio[stem].mean().item())
                raise ValueError(f"NaN in denormalized separated_audio for stem {stem}")


        return separated_audio

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Separator":
        """
        Instantiates a Separator model from a DictConfig.

        This class method is responsible for parsing the configuration and
        constructing a `SeparatorConfig` object, which is then used to
        initialize the `Separator` model.

        Args:
            cfg (DictConfig): A DictConfig object containing the model configuration.

        Returns:
            Separator: An instance of the Separator model.
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

        # Instantiate band_specs first, as it requires n_fft and fs
        bandsplit_type = model_cfg.bandsplit.band_specs.bandsplit_type
        n_bands = model_cfg.bandsplit.band_specs.n_bands
        band_specs: BandsplitSpecification = get_bandsplit_specs_factory(
            bandsplit_type=bandsplit_type,
            n_fft=stft_config.n_fft,
            fs=fs,
            n_bands=n_bands,
            drop_dc_band=drop_dc_band # Pass drop_dc_band to factory
        )

        # Instantiate bandsplit_config using hydra.utils.instantiate
        # Manually create bandsplit_config, passing the instantiated band_specs
        # Manually create bandsplit_config, passing the instantiated band_specs
        # Manually create bandsplit_config, passing the instantiated band_specs
        bandsplit_config: BandsplitModuleConfig = hydra.utils.instantiate(
            model_cfg.bandsplit,
            band_specs=band_specs, # Pass the already instantiated band_specs
            _recursive_=False # Prevent re-instantiation of band_specs
        )

        # Create the main SeparatorConfig instance
        separator_config: SeparatorConfig = SeparatorConfig(
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
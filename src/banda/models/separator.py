#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from banda.data.batch_types import SeparationBatch, FixedStemSeparationBatch
from banda.models.spectral import SpectralComponent
from banda.models.components.bandsplit import BandSplitModule
from banda.models.components.mask_estimation import MaskEstimationModule, OverlappingMaskEstimationModule
from banda.models.components.time_frequency_model import SeqBandModellingModule, TransformerTimeFreqModule, ConvolutionalTimeFreqModule
from banda.models.utils import VDBO_STEMS # Example default stems


class Separator(SpectralComponent):
    """
    A fixed-stem Bandit-like source separation model.

    This model takes a mixture audio as input and outputs separated sources
    using a bandsplit approach with a time-frequency model and mask estimation.
    """

    def __init__(
        self,
        stems: List[str] = VDBO_STEMS, # Default to VDBO stems
        band_specs: List[Tuple[float, float]] = None, # Frequency band specifications
        fs: int = 44100,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        power: Optional[int] = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        # BandsplitModule parameters
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
        in_channel: int = 2, # Stereo audio
        # Time-Frequency Model parameters
        tf_model_type: str = "rnn", # "rnn", "transformer", or "conv"
        n_tf_modules: int = 12,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        tf_dropout: float = 0.0,
        # Mask Estimation parameters
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        overlapping_band: bool = False,
        use_freq_weights: bool = False,
        # Note: freq_weights and n_freq will be handled internally by MaskEstimationModule
    ) -> None:
        """
        Initializes the Separator model.
        """
        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            center=center,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided,
        )

        self.stems = stems
        self.band_specs = band_specs
        self.fs = fs
        self.n_fft = n_fft
        self.in_channel = in_channel

        # 1. Bandsplit Module
        if self.band_specs is None:
            # Default band specs if not provided, e.g., full band
            self.band_specs = [(0, n_fft // 2 + 1)] # Full frequency range
            # For a more sophisticated default, you'd use a BandSplitSpecification like in bandit/wrapper.py

        self.band_split = BandSplitModule(
            band_specs=self.band_specs,
            emb_dim=emb_dim,
            in_channel=in_channel,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
        )

        # 2. Time-Frequency Model
        if tf_model_type == "rnn":
            self.tf_model = SeqBandModellingModule(
                n_modules=n_tf_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
            )
        elif tf_model_type == "transformer":
            self.tf_model = TransformerTimeFreqModule(
                n_modules=n_tf_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim, # Not used, but for compatibility
                bidirectional=bidirectional, # Not used, but for compatibility
                dropout=tf_dropout,
            )
        elif tf_model_type == "conv":
            self.tf_model = ConvolutionalTimeFreqModule(
                n_modules=n_tf_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim, # Not used, but for compatibility
                bidirectional=bidirectional, # Not used, but for compatibility
                dropout=tf_dropout,
            )
        else:
            raise ValueError(f"Unknown tf_model_type: {tf_model_type}")

        # 3. Mask Estimation Module
        mask_estim_cls = OverlappingMaskEstimationModule if overlapping_band else MaskEstimationModule
        self.mask_estim = nn.ModuleDict(
            {
                stem: mask_estim_cls(
                    band_specs=self.band_specs,
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    in_channel=in_channel,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    complex_mask=complex_mask,
                    # Parameters specific to OverlappingMaskEstimationModule
                    freq_weights=None, # Will be set by a BandSplitSpecification if used
                    n_freq=n_fft // 2 + 1, # Total frequency bins
                    use_freq_weights=use_freq_weights,
                )
                for stem in stems
            }
        )

    def forward(self, batch: SeparationBatch) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the separator model.

        Args:
            batch (SeparationBatch): The input batch containing the mixture audio.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of separated source audio tensors.
                                     Each tensor has shape: (batch_size, channels, samples)
        """
        # Extract mixture from the batch based on its type
        if isinstance(batch, FixedStemSeparationBatch):
            mixture_audio = batch.audio.mixture
        else:
            raise NotImplementedError(f"Model forward pass not implemented for batch type: {type(batch)}")

        # Store original audio length for ISTFT
        original_audio_length = mixture_audio.shape[-1]

        # 1. STFT: Convert audio to complex spectrogram
        # Output shape: (batch_size, channels, freq_bins, time_frames, 2)
        mixture_spec = self.stft(mixture_audio)

        # 2. Bandsplit: Process each frequency band
        # Output shape: (batch, n_bands, n_time, emb_dim)
        band_features = self.band_split(mixture_spec)

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
            # Need to expand mask to have the real/imaginary dimension
            masked_spec = mixture_spec * mask.unsqueeze(-1) # Add a dimension for real/imaginary

            # ISTFT: Convert masked spectrogram back to audio
            separated_audio[stem] = self.inverse(masked_spec, original_audio_length)

        return separated_audio
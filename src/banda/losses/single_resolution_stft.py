import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T # Keep for potential future use or if other parts rely on it
from typing import Dict, Optional
from omegaconf import DictConfig

from banda.losses.l1_snr_utils import calculate_l1_snr


class SingleResolutionSTFTLoss(nn.Module):
    """
    Computes the STFT loss for a single resolution, implementing a Signal-to-Noise P-Norm Ratio loss
    on the real and imaginary parts of the spectrograms, similar to the Bandit model.
    """
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        # Removed p, scale_invariant, take_log, eps as they are handled by calculate_l1_snr
        # Removed reduction as it will be handled by SeparationLossHandler
    ) -> None:
        """
        Args:
            n_fft (int): n_fft value for the STFT resolution.
            hop_length (int): hop_length value for the STFT resolution.
            win_length (int): win_length value for the STFT resolution.
            window_fn (str): Name of the window function to use (e.g., "hann_window").
            wkwargs (Optional[Dict]): Keyword arguments for the window function.
        """
        super().__init__()
        
        # Store STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = True # Default for torchaudio.transforms.Spectrogram
        self.pad_mode = "reflect" # Default for torchaudio.transforms.Spectrogram
        self.normalized = False # Default for torchaudio.transforms.Spectrogram
        self.onesided = True # Default for torchaudio.transforms.Spectrogram

        # Create the window tensor and register it as a buffer
        # wkwargs is not directly used here as torch.stft does not take wkwargs
        # If specific window kwargs are needed, they should be handled when creating the window tensor
        window = torch.__dict__[window_fn](win_length)
        self.register_buffer("window", window)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the single-resolution STFT loss.

        Args:
            prediction (torch.Tensor): Predicted audio waveform. Shape: (batch, channels, samples)
            target (torch.Tensor): Ground truth audio waveform. Shape: (batch, channels, samples)

        Returns:
            torch.Tensor: The calculated single-resolution STFT loss.
        """
        batch_size, num_channels, num_samples = prediction.shape

        # Reshape to (batch_size * num_channels, num_samples) for STFT
        prediction_reshaped = prediction.view(batch_size * num_channels, num_samples)
        target_reshaped = target.view(batch_size * num_channels, num_samples)

        # Ensure the window is on the same device as the input prediction tensor
        window_on_device = self.window.to(prediction_reshaped.device)

        # Use torch.stft directly
        pred_spec = torch.stft(
            prediction_reshaped, # Use reshaped tensor
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_on_device, # Pass the window explicitly moved to device
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        target_spec = torch.stft(
            target_reshaped, # Use reshaped tensor
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_on_device, # Pass the window explicitly moved to device
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

        # Reshape spectrograms back to (batch_size, num_channels, freq, time)
        # STFT output is (batch_size * num_channels, freq, time)
        pred_spec = pred_spec.view(batch_size, num_channels, pred_spec.shape[-2], pred_spec.shape[-1])
        target_spec = target_spec.view(batch_size, num_channels, target_spec.shape[-2], target_spec.shape[-1])

        # Convert complex spectrograms to real/imaginary parts
        # Shape: (batch, channels, freq, time, 2) where last dim is [real, imag]
        pred_spec_real_imag = torch.view_as_real(pred_spec)
        target_spec_real_imag = torch.view_as_real(target_spec)

        # Calculate L1 SNR using the utility function
        loss = calculate_l1_snr(
            pred_spec_real_imag,
            target_spec_real_imag,
            scale_invariant=True, # STFT loss is typically scale-invariant
            take_log=True,
            eps=1e-8,
        )

        return loss

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Instantiates a SingleResolutionSTFTLoss from a DictConfig.
        """
        n_fft = config.n_fft
        hop_length = config.hop_length
        win_length = config.win_length
        window_fn = config.get("window_fn", "hann_window")
        wkwargs = config.get("wkwargs", None)

        return cls(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
        )
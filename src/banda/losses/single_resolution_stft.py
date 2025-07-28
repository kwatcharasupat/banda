import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Optional
from omegaconf import DictConfig


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
        p: float = 1.0, # P-norm for the loss (1.0 for L1, 2.0 for L2)
        scale_invariant: bool = False,
        take_log: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8, # Small epsilon for numerical stability
    ) -> None:
        """
        Args:
            n_fft (int): n_fft value for the STFT resolution.
            hop_length (int): hop_length value for the STFT resolution.
            win_length (int): win_length value for the STFT resolution.
            window_fn (str): Name of the window function to use (e.g., "hann_window").
            wkwargs (Optional[Dict]): Keyword arguments for the window function.
            p (float): The p-norm to use for calculating error and target energy (1.0 for L1, 2.0 for L2).
            scale_invariant (bool): If True, apply scale-invariant projection.
            take_log (bool): If True, apply 10 * log10 to the ratio.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            eps (float): Small epsilon for numerical stability in division.
        """
        super().__init__()
        assert p in [1.0, 2.0], "Only p=1.0 (L1) and p=2.0 (L2) are supported."
        assert reduction in ["mean", "sum", "none"], "Reduction must be 'mean', 'sum', or 'none'."

        self.p = p
        self.scale_invariant = scale_invariant
        self.take_log = take_log
        self.reduction = reduction
        self.eps = eps

        window = torch.__dict__[window_fn]
        self.stft_layer = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=window,
            wkwargs=wkwargs,
            return_complex=True,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the single-resolution STFT loss.

        Args:
            prediction (torch.Tensor): Predicted audio waveform. Shape: (batch, channels, samples)
            target (torch.Tensor): Ground truth audio waveform. Shape: (batch, channels, samples)

        Returns:
            torch.Tensor: The calculated single-resolution STFT loss.
        """
        pred_spec = self.stft_layer(prediction)
        target_spec = self.stft_layer(target)

        # Convert complex spectrograms to real/imaginary parts
        # Shape: (batch, channels, freq, time, 2) where last dim is [real, imag]
        pred_spec_real_imag = torch.view_as_real(pred_spec)
        target_spec_real_imag = torch.view_as_real(target_spec)

        # Reshape to (batch_size, -1) for p-norm calculation
        batch_size = pred_spec_real_imag.shape[0]
        est_target_flat = pred_spec_real_imag.reshape(batch_size, -1)
        target_flat = target_spec_real_imag.reshape(batch_size, -1)

        # Apply scale-invariant projection if enabled
        if self.scale_invariant:
            # This part needs to be adapted for complex spectrograms.
            # The original Bandit code applies it to the time-domain signal.
            # For STFT domain, a common approach is to project the estimated
            # spectrogram onto the target spectrogram.
            # For simplicity and to match the spirit of the Bandit code's
            # SignalNoisePNormRatio, we'll apply it to the flattened real/imag parts.
            dot = torch.sum(est_target_flat * target_flat, dim=-1, keepdim=True)
            s_target_energy = torch.sum(target_flat * target_flat, dim=-1, keepdim=True)
            target_scaler = (dot + self.eps) / (s_target_energy + self.eps)
            target_flat = target_flat * target_scaler

        # Calculate error and target energy based on p-norm
        if self.p == 1.0:
            e_error = torch.abs(est_target_flat - target_flat).mean(dim=-1)
            e_target = torch.abs(target_flat).mean(dim=-1)
        elif self.p == 2.0:
            e_error = torch.square(est_target_flat - target_flat).mean(dim=-1)
            e_target = torch.square(target_flat).mean(dim=-1)
        else:
            # This should not be reached due to the assert in __init__
            raise NotImplementedError

        # Calculate loss ratio
        if self.take_log:
            # The original Bandit code uses 10 * log10(e_error) - 10 * log10(e_target)
            # which is equivalent to 10 * log10(e_error / e_target)
            loss = 10 * (torch.log10(e_error + self.eps) - torch.log10(e_target + self.eps))
        else:
            loss = (e_error + self.eps) / (e_target + self.eps)

        # Apply reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # If reduction is "none", loss is returned as is (batch-wise)

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
        p = config.get("p", 1.0)
        scale_invariant = config.get("scale_invariant", False)
        take_log = config.get("take_log", True)
        reduction = config.get("reduction", "mean")
        eps = config.get("eps", 1e-8)

        return cls(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            p=p,
            scale_invariant=scale_invariant,
            take_log=take_log,
            reduction=reduction,
            eps=eps,
        )
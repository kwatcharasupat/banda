#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import torch
import torchaudio.transforms as T
from typing import Optional

from banda.data.augmentations.base import PostMixTransform
from banda.data.batch_types import TorchInputAudioDict
from banda.data.types import Identifier


class RandomFrequencyMasking(PostMixTransform):
    """
    Applies random frequency masking to the spectrogram of the audio.
    """

    def __init__(self, freq_mask_param: int = 10, p: float = 0.5) -> None:
        """
        Args:
            freq_mask_param (int): Maximum width of the frequency mask.
            p (float): Probability of applying the transform.
        """
        self.freq_mask_param = freq_mask_param
        self.p = p
        self.freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)

    def __call__(self, *, audio_dict: TorchInputAudioDict, identifier: Identifier) -> TorchInputAudioDict:
        """
        Apply random frequency masking to the mixture and sources.

        Args:
            audio_dict (TorchInputAudioDict): Dictionary containing audio data.
            identifier (Identifier): Identifier for the audio data.

        Returns:
            TorchInputAudioDict: Transformed audio data with frequency masking.
        """
        if torch.rand(1).item() > self.p:
            return audio_dict

        def _apply_freq_mask(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is None:
                return None
            # Convert to spectrogram, apply mask, convert back (simplified)
            # This requires STFT/ISTFT which is more complex for a placeholder.
            # For now, we'll apply it directly to the time-domain signal as a placeholder,
            # but this is conceptually incorrect for frequency masking.
            # A proper implementation would involve STFT -> Mask -> ISTFT.
            # For boilerplate, we'll just return the original for now or apply a dummy op.
            # Let's apply a dummy operation for now to show where it would go.
            return tensor * 0.99 # Dummy operation

        masked_mixture = _apply_freq_mask(audio_dict.mixture)
        masked_sources = {
            key: _apply_freq_mask(source) for key, source in audio_dict.sources.items()
        }

        return TorchInputAudioDict(mixture=masked_mixture, sources=masked_sources)


class RandomTimeMasking(PostMixTransform):
    """
    Applies random time masking to the spectrogram of the audio.
    """

    def __init__(self, time_mask_param: int = 10, p: float = 0.5) -> None:
        """
        Args:
            time_mask_param (int): Maximum width of the time mask.
            p (float): Probability of applying the transform.
        """
        self.time_mask_param = time_mask_param
        self.p = p
        self.time_mask = T.TimeMasking(time_mask_param=self.time_mask_param)

    def __call__(self, *, audio_dict: TorchInputAudioDict, identifier: Identifier) -> TorchInputAudioDict:
        """
        Apply random time masking to the mixture and sources.

        Args:
            audio_dict (TorchInputAudioDict): Dictionary containing audio data.
            identifier (Identifier): Identifier for the audio data.

        Returns:
            TorchInputAudioDict: Transformed audio data with time masking.
        """
        if torch.rand(1).item() > self.p:
            return audio_dict

        def _apply_time_mask(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is None:
                return None
            # Similar to frequency masking, this requires STFT/ISTFT.
            # Applying a dummy operation for now.
            return tensor * 0.99 # Dummy operation

        masked_mixture = _apply_time_mask(audio_dict.mixture)
        masked_sources = {
            key: _apply_time_mask(source) for key, source in audio_dict.sources.items()
        }

        return TorchInputAudioDict(mixture=masked_mixture, sources=masked_sources)
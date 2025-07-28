#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import structlog

logger = structlog.get_logger()

def load_audio(file_path: str, target_fs: int = 44100, target_channels: int = 2) -> np.ndarray:
    """
    Loads an audio file, resamples if necessary, and ensures consistent channel count.
    This function is designed for .wav files.
    """
    audio, fs = sf.read(file_path, always_2d=True)
    audio = torch.from_numpy(audio).float().T # Convert to (channels, samples) and to tensor

    if fs != target_fs:
        logger.info(f"Resampling {file_path} from {fs} Hz to {target_fs} Hz.")
        resampler = T.Resample(orig_freq=fs, new_freq=target_fs)
        audio = resampler(audio)

    if audio.shape[0] != target_channels:
        if audio.shape[0] == 1 and target_channels == 2:
            # Convert mono to stereo by duplicating channel
            audio = audio.repeat(target_channels, 1)
        elif audio.shape[0] == 2 and target_channels == 1:
            # Convert stereo to mono by averaging channels
            audio = torch.mean(audio, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported channel conversion for {file_path}: {audio.shape[0]} to {target_channels}.")
    
    return audio.numpy() # Convert back to numpy array
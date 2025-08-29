import glob
import math
import os
import subprocess
from tempfile import TemporaryFile
from typing import Dict, List, Optional
import numpy as np
import torch
import torchaudio as ta
from tqdm.contrib.concurrent import process_map
from pydantic import BaseModel

import yaml


def round_samples(seconds: float, fs: int, hop_size: int, downsample: int) -> int:
    """
    Calculate the number of samples for a given duration, considering hop size and downsampling.

    Args:
        seconds (float): Duration in seconds.
        fs (int): Sampling rate in Hz.
        hop_size (int): Hop size in samples.
        downsample (int): Downsampling factor.

    Returns:
        int: Number of samples.
    """
    n_frames = math.ceil(seconds * fs / hop_size) + 1
    n_frames_down = math.ceil(n_frames / downsample)
    n_frames = n_frames_down * downsample
    n_samples = (n_frames - 1) * hop_size

    return int(n_samples)


def ffmpeg_resample(
    input_path: str,
    output_path: str,
    fs: int = 44100,
) -> None:
    """
    Resample an audio file to a specified sample rate using ffmpeg.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the output resampled audio file.
        fs (int, optional): Target sample rate for the audio. Defaults to 44100 Hz.
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ar",
        str(fs),
        "-f",
        "wav",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def to_npz(
    raw_path: str, npz_path: str, fs: int = 44100, allow_resampling: bool = False
) -> None:
    """
    Convert an audio file to a NumPy `.npz` file with the specified sample rate.

    Args:
        raw_path (str): Path to the input raw audio file.
        npz_path (str): Path to save the output `.npz` file.
        fs (int, optional): Target sample rate for the audio. Defaults to 44100 Hz.
        allow_resampling (bool, optional): Whether to allow resampling if the sample rate does not match. Defaults to False.

    Raises:
        ValueError: If resampling is required but not allowed.
        AssertionError: If the sample rate or tensor dimensions are invalid.
    """
    metadata = ta.info(raw_path)

    fs_ = metadata.sample_rate
    if fs_ != fs:
        if not allow_resampling:
            raise ValueError(
                f"Expected {fs}, got {fs_}. Use allow_resampling=True to resample."
            )
        with TemporaryFile(suffix=".wav") as tmp_file:
            ffmpeg_resample(raw_path, tmp_file.name, fs=fs)
            tmp_file.flush()
            tmp_file.seek(0)
            wav, fs_ = ta.load(tmp_file.name)
    else:
        wav, fs_ = ta.load(raw_path)

    assert fs_ == fs, f"Expected {fs}, got {fs_}"

    save_npz(
        wav=wav,
        fs=fs_,
        npz_path=npz_path,
    )


def save_npz(
    *,
    wav: torch.Tensor,
    fs: int,
    npz_path: str,
) -> None:
    """
    Save a NumPy array to a `.npz` file with the specified sample rate.

    Args:
        wav (torch.Tensor): Audio data as a PyTorch tensor of shape (channels, samples).
        fs (int): Sample rate of the audio.
        npz_path (str): Path to save the output `.npz` file.

    Raises:
        AssertionError: If the input tensor is not 2D.
    """
    assert wav.ndim == 2, f"Expected 2D tensor, got {wav.ndim}D tensor"

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(
        npz_path,
        audio=wav.numpy(),
        fs=fs,
    )


def npzify_dataset(
    *,
    raw_data_root: str,
    processed_data_root: str,
    fs: int = 44100,
    max_workers: Optional[int] = None,
    chunksize: Optional[int] = 1,
    commit: bool = False,
    verbose: bool = True,
) -> None:
    """
    Convert a dataset of audio files to `.npz` format.

    Args:
        raw_data_root (str): Root directory of raw audio files.
        processed_data_root (str): Root directory to save processed `.npz` files.
        fs (int, optional): Target sample rate for the audio. Defaults to 44100 Hz.
        max_workers (Optional[int], optional): Maximum number of workers for parallel processing. Defaults to None.
        chunksize (Optional[int], optional): Chunk size for parallel processing. Defaults to 1.
        commit (bool, optional): Whether to commit the changes. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    """
    raw_data_root = os.path.expandvars(raw_data_root)
    processed_data_root = os.path.expandvars(processed_data_root)
    os.makedirs(processed_data_root, exist_ok=True)
    wav_files = glob.glob(os.path.join(raw_data_root, "**", "*.wav"), recursive=True)
    npz_files = [
        os.path.join(
            processed_data_root,
            os.path.relpath(wav_file, raw_data_root).replace(".wav", ".npz"),
        )
        for wav_file in wav_files
    ]
    fs_ = [fs] * len(wav_files)

    if verbose:
        print(f"Found {len(wav_files)} wav files")
        wav_file = wav_files[0]
        npz_file = npz_files[0]
        print(f"First wav file: {wav_file}")
        print(f"First npz file: {npz_file}")

    if not commit and verbose:
        commit_ = input("Do you want to proceed? (y/n) [default: n]: ")
        commit = commit_.lower() == "y"

    if commit:
        process_map(
            to_npz,
            wav_files,
            npz_files,
            fs_,
            max_workers=max_workers,
            chunksize=chunksize,
        )


def pad_and_mix(
    *,
    stem_list: List[torch.Tensor],
    n_samples: int,
    strict: bool = True,
) -> torch.Tensor:
    """
    Pad and mix a list of audio stems to a specified number of samples.

    Args:
        stem_list (List[torch.Tensor]): List of audio stems as PyTorch tensors.
        n_samples (int): Number of samples to pad or truncate to.
        strict (bool, optional): Whether to raise an error if a stem exceeds `n_samples`. Defaults to True.

    Returns:
        torch.Tensor: Mixed audio stem as a PyTorch tensor.
    """
    padded_stem_list = []
    for stem in stem_list:
        _, n_samples_ = stem.shape
        if n_samples_ < n_samples:
            pad_size = n_samples - n_samples_
            padded_stem = torch.nn.functional.pad(stem, (0, pad_size))
        elif n_samples_ > n_samples:
            if strict:
                raise ValueError(
                    f"Expected {n_samples} samples, got {n_samples_} samples"
                )
            padded_stem = stem[:, :n_samples]
        else:
            padded_stem = stem

        padded_stem_list.append(padded_stem)

    mixed_stem = sum(padded_stem_list, torch.tensor(0.0))

    return mixed_stem


class StemAliasConfig(BaseModel):
    """
    Pydantic model for stem alias configuration.

    Attributes:
        stem_alias (Dict[str, str]): Mapping of stem names to their aliases.
    """

    stem_alias: Dict[str, str]


class BaseStemAlias:
    """
    Handle stem aliasing using a Pydantic model for validation.

    Args:
        yaml_path (str): Path to the YAML file containing stem alias mappings.
    """

    def __init__(self, yaml_path: str) -> None:
        """
        Initialize the BaseStemAlias class.

        Args:
            yaml_path (str): Path to the YAML file containing stem alias mappings.
        """

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        self.config = StemAliasConfig(stem_alias=data)

    def get(
        self,
        stem_name: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the alias for a stem name.

        Args:
            stem_name (str): Name of the stem.

        Returns:
            Optional[str]: Alias for the stem name, or None if not found.
        """
        return self.config.stem_alias.get(stem_name, default)

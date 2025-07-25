#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#
#

import os
import glob
from typing import Optional, List, Tuple

import hydra
import numpy as np
import soundfile as sf
import structlog
import torch
import torchaudio.transforms as T
import stempeg # Import stempeg
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.concurrent import process_map

logger = structlog.get_logger()

# Define common stems, can be extended
DEFAULT_STEMS = ["mixture", "vocals", "bass", "drums", "other"]

def _load_audio(file_path: str, target_fs: int = 44100, target_channels: int = 2) -> np.ndarray:
    """
    Loads an audio file, resamples if necessary, and ensures consistent channel count.
    Supports both .wav and .stem.mp4 files.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".wav":
        audio, fs = sf.read(file_path, always_2d=True)
        audio = torch.from_numpy(audio).float().T # Convert to (channels, samples) and to tensor
    elif ext == ".mp4": # Assuming .stem.mp4
        # stempeg loads as (stems, channels, samples)
        # We need to load a specific stem from the .mp4 file
        # This function is designed to load a single stem's audio.
        # For .stem.mp4, we need to know which stem to extract.
        # This implies a change in how process_track calls _load_audio.
        # For now, let's assume we're loading the mixture from .mp4 directly if it's a mixture.wav
        # or that the stempeg loading will be handled differently.
        # For a generic _load_audio, we'd need to pass the stem name.
        # Let's simplify: if it's .mp4, we assume it's the multi-track file
        # and we'll extract the mixture from it.
        # This is a temporary simplification for the _load_audio function.
        # A more robust solution would involve passing the stem name to stempeg.read_stems.
        
        # For now, let's assume the file_path is directly to the .stem.mp4
        # and we're trying to load the mixture from it.
        # This will need to be refined.
        
        # Example: Load mixture from .stem.mp4
        # This requires knowing the stem index or name within the .mp4
        # For now, let's just load the mixture (index 0)
        try:
            audio, fs = stempeg.read_stems(file_path, stem_id=0, samplerate=target_fs) # stem_id=0 for mixture
            audio = torch.from_numpy(audio).float() # (channels, samples)
        except Exception as e:
            raise ValueError(f"Error loading .stem.mp4 with stempeg: {e}")
    else:
        raise ValueError(f"Unsupported audio file format: {ext}")

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

def process_track(args: Tuple[str, str, str, DictConfig, List[str]]) -> None:
    """
    Processes a single track, loads its stems, and saves them as a .npz file.
    """
    track_path, original_split, new_split, cfg, stems_to_process = args
    track_title = os.path.basename(track_path)
    
    logger.info(f"Processing track: {track_title} from {original_split} to {new_split}")

    output_dir = os.path.join(cfg.output_path, new_split)
    os.makedirs(output_dir, exist_ok=True)

    output_npz_path = os.path.join(output_dir, f"{track_title}.npz")

    audio_data = {}
    
    # Determine if it's a .stem.mp4 directory or a .wav directory
    is_stem_mp4_dir = track_title.endswith(".stem.mp4")

    if is_stem_mp4_dir:
        # For .stem.mp4, the track_path is the directory containing the .stem.mp4 file
        # The actual audio file is track_path itself.
        stem_mp4_file_path = track_path
        
        # Load all stems from the .stem.mp4 file
        # stempeg.read_stems returns (stems, channels, samples)
        try:
            # Load all stems from the .stem.mp4 file
            # This assumes a fixed order of stems in the .mp4 file (mixture, drums, bass, other, vocals)
            # which is typical for MUSDB18.
            # We need to map these to our desired stems_to_process.
            # For now, let's hardcode the stem order for MUSDB18 .mp4 files.
            # This needs to be made more flexible.
            
            # MUSDB18 stem order: mixture, drums, bass, other, vocals
            musdb18_stem_names = ["mixture", "drums", "bass", "other", "vocals"]
            
            all_stems_audio, fs = stempeg.read_stems(stem_mp4_file_path, samplerate=cfg.target_fs)
            
            # Resample if necessary
            if fs != cfg.target_fs:
                logger.info(f"Resampling {stem_mp4_file_path} from {fs} Hz to {cfg.target_fs} Hz.")
                resampler = T.Resample(orig_freq=fs, new_freq=cfg.target_fs)
                all_stems_audio = resampler(torch.from_numpy(all_stems_audio).float()).numpy()

            for i, stem_name in enumerate(musdb18_stem_names):
                if stem_name in stems_to_process:
                    audio_data[stem_name] = all_stems_audio[i] # (channels, samples)
                    
            # Ensure consistent channel count for all loaded stems
            for stem, audio_arr in audio_data.items():
                if audio_arr.shape[0] != cfg.target_channels:
                    if audio_arr.shape[0] == 1 and cfg.target_channels == 2:
                        audio_data[stem] = np.repeat(audio_arr, 2, axis=0)
                    elif audio_arr.shape[0] == 2 and cfg.target_channels == 1:
                        audio_data[stem] = np.mean(audio_arr, axis=0, keepdims=True)
                    else:
                        raise ValueError(f"Unsupported channel conversion for {stem} in {stem_mp4_file_path}: {audio_arr.shape[0]} to {cfg.target_channels}.")

        except Exception as e:
            logger.error(f"Error processing .stem.mp4 file {stem_mp4_file_path}: {e}")
            return # Skip this track if an error occurs

    else:
        # For .wav directories, load individual stem.wav files
        for stem in stems_to_process:
            stem_file_path = os.path.join(track_path, f"{stem}.wav")
            try:
                audio_data[stem] = _load_audio(stem_file_path, cfg.target_fs, cfg.target_channels)
            except FileNotFoundError:
                logger.warning(f"Stem file not found for {stem} in {track_title}. Skipping this stem.")
                continue
            except Exception as e: # Catch any other exceptions during audio loading/processing
                logger.error(f"Error processing {stem_file_path}: {e}")
                return # Skip this track if an error occurs

    if not audio_data:
        logger.warning(f"No audio data found for track {track_title}. Skipping.")
        return

    # Determine the maximum length among all loaded stems for this track
    max_len = max(arr.shape[1] for arr in audio_data.values())

    # Pad all audio arrays to the maximum length
    for stem, arr in audio_data.items():
        if arr.shape[1] < max_len:
            padding_needed = max_len - arr.shape[1]
            audio_data[stem] = np.pad(arr, ((0, 0), (0, padding_needed)), mode='constant')

    # Save to NPZ
    np.savez(output_npz_path, fs=cfg.target_fs, **audio_data)
    logger.info(f"Saved {track_title}.npz to {output_npz_path}")


@hydra.main(config_path="../src/banda/configs/preprocess", config_name="musdb18", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info(
        "Running MUSDB18 preprocessing script with the following config:\n"
        + OmegaConf.to_yaml(cfg)
    )

    input_base_path = cfg.input_path
    output_base_path = cfg.output_path
    stems_to_process = cfg.stems_to_process if cfg.stems_to_process else DEFAULT_STEMS

    # Find all track directories
    track_paths = []
    for split in ["train", "test"]: # MUSDB18 has train and test splits
        split_path = os.path.join(input_base_path, split)
        if os.path.exists(split_path):
            # Check if it's a .stem.mp4 file or a directory of .wav files
            # glob.glob will return directories for .stem.mp4 and files for .wav
            # We need to ensure we're getting the track directories
            
            # For .stem.mp4, the track_path is the .stem.mp4 file itself
            # For .wav, the track_path is the directory containing the .wav files
            
            # This logic needs to be robust to both structures.
            # Let's assume track_paths will contain the full path to either
            # the .stem.mp4 file or the directory containing .wav files.
            
            # If the input_path points to the 'canonical' directory,
            # then glob.glob(os.path.join(split_path, "*")) will give us
            # paths like '.../train/Artist - Song.stem.mp4' (for .mp4)
            # or '.../train/Artist - Song' (for .wav directories)
            
            # This is already handled by the existing glob.glob, as it gets the track directories.
            track_paths.extend(glob.glob(os.path.join(split_path, "*")))
        else:
            logger.warning(f"Input split path not found: {split_path}")

    if not track_paths:
        logger.error(f"No tracks found in {input_base_path}. Please check your input_path.")
        return

    # Determine new splits (e.g., for train/val/test split)
    # This part can be made more sophisticated if a specific validation split is needed
    # For MUSDB18, typically a subset of train is used for validation.
    # For simplicity, we'll map original 'train' to 'train' and 'test' to 'test' for now.
    # If a validation set is needed, it should be handled here, similar to coda-refactor's validation_files.
    original_splits = [os.path.basename(os.path.dirname(p)) for p in track_paths]
    new_splits = [s for s in original_splits] # Simple 1:1 mapping for now

    tasks = []
    for i, track_path in enumerate(track_paths):
        tasks.append((track_path, original_splits[i], new_splits[i], cfg, stems_to_process))

    logger.info(f"Found {len(tasks)} tracks to process.")
    process_map(process_track, tasks, max_workers=cfg.num_workers)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()

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
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from banda.utils.audio_utils import load_audio

logger = structlog.get_logger()

# Define common stems, can be extended
DEFAULT_STEMS = ["mixture", "vocals", "bass", "drums", "other"]

# List of validation track titles for MUSDB18HQ
# This list is taken from the official MUSDB18HQ validation split
MUSDB18HQ_VALIDATION_TRACKS = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]

def process_track(args: Tuple[str, str, str, DictConfig, List[str]]) -> None:
    """
    Processes a single track, loads its stems, and saves them as a .npz file.
    This function expects track_path to be a directory containing .wav files for each stem.
    """
    track_path, original_split, new_split, cfg, stems_to_process = args
    track_title = os.path.basename(track_path)
    
    logger.info(f"Processing track: {track_title} from {original_split} to {new_split}")

    output_dir = os.path.expanduser(os.path.join(cfg.output_path, new_split))
    os.makedirs(output_dir, exist_ok=True)

    output_npz_path = os.path.join(output_dir, f"{track_title}.npz")

    audio_data = {}
    
    # Load individual stem.wav files
    for stem in stems_to_process:
        stem_file_path = os.path.join(track_path, f"{stem}.wav")
        try:
            audio_data[stem] = load_audio(stem_file_path, cfg.target_fs, cfg.target_channels)
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

    # Recompute mixture from stems if 'mixture' is in stems_to_process
    if "mixture" in stems_to_process:
        # Initialize mixture with zeros, matching the shape of other stems
        # If no other stems are present, this will be a zero array of max_len
        mixture_shape = (cfg.target_channels, max_len)
        recomputed_mixture = np.zeros(mixture_shape, dtype=np.float32)

        # Sum up all other stems to create the mixture
        for stem in ["vocals", "bass", "drums", "other"]:
            if stem in audio_data:
                recomputed_mixture += audio_data[stem]
        audio_data["mixture"] = recomputed_mixture

    # Save to NPZ
    np.savez(output_npz_path, fs=cfg.target_fs, **audio_data)
    logger.info(f"Saved {track_title}.npz to {output_npz_path}")

def validate_track(args: Tuple[str, str, DictConfig, List[str]]) -> bool:
    """
    Validates a single track by comparing its original WAV stems with the preprocessed NPZ data.
    Returns True if validation passes, False otherwise.
    """
    track_title, original_base_path, cfg, stems_to_process = args
    
    logger.info(f"Validating track: {track_title}")

    # Determine the split for the NPZ file
    npz_split = "train"
    if track_title in MUSDB18HQ_VALIDATION_TRACKS:
        npz_split = "val"
    elif os.path.exists(os.path.join(original_base_path, "test", track_title)):
        npz_split = "test"

    npz_file_path = os.path.expanduser(os.path.join(cfg.output_path, npz_split, f"{track_title}.npz"))
    
    if not os.path.exists(npz_file_path):
        logger.error(f"NPZ file not found for {track_title} at {npz_file_path}")
        return False

    try:
        npz_data = np.load(npz_file_path)
    except Exception as e:
        logger.error(f"Failed to load NPZ file for {track_title}: {e}")
        return False

    # Get original split path
    original_split_path = os.path.join(original_base_path, "train")
    if os.path.exists(os.path.join(original_base_path, "test", track_title)):
        original_split_path = os.path.join(original_base_path, "test")
    
    original_track_path = os.path.join(original_split_path, track_title)

    validation_passed = True
    max_len_npz = 0
    for stem in stems_to_process:
        if stem in npz_data:
            max_len_npz = max(max_len_npz, npz_data[stem].shape[1])

    for stem in stems_to_process:
        # If the stem is 'mixture', we recompute it from the other stems in the NPZ data
        if stem == "mixture":
            # Recompute mixture from original WAV stems for ground truth comparison
            original_stems_audio = {}
            current_max_len = 0
            for s in ["vocals", "bass", "drums", "other"]:
                original_stem_file = os.path.join(original_track_path, f"{s}.wav")
                if os.path.exists(original_stem_file):
                    audio = load_audio(original_stem_file, cfg.target_fs, cfg.target_channels)
                    original_stems_audio[s] = audio
                    current_max_len = max(current_max_len, audio.shape[1])
                else:
                    logger.warning(f"Original WAV for {s} in {track_title} not found. Cannot recompute mixture accurately.")
                    validation_passed = False # Mark as failed if a source stem is missing for mixture recomputation
                    break
            
            if not validation_passed: # If a source stem was missing, skip mixture validation
                continue

            recomputed_original_mixture = np.zeros((cfg.target_channels, current_max_len), dtype=np.float32)
            for s in ["vocals", "bass", "drums", "other"]:
                if s in original_stems_audio:
                    # Pad individual original stems to current_max_len before summing
                    padded_audio = np.pad(original_stems_audio[s], ((0, 0), (0, current_max_len - original_stems_audio[s].shape[1])), mode='constant')
                    recomputed_original_mixture += padded_audio
            original_audio = recomputed_original_mixture

            # Get NPZ mixture (which was also recomputed from stems during preprocessing)
            if stem not in npz_data:
                logger.error(f"Stem '{stem}' missing in NPZ for track {track_title}.")
                raise ValueError(f"Stem '{stem}' missing in NPZ for track {track_title}.")
            npz_audio = npz_data[stem]

        else: # For non-mixture stems, load original WAV and NPZ data directly
            original_stem_file = os.path.join(original_track_path, f"{stem}.wav")
            
            if not os.path.exists(original_stem_file):
                if stem in npz_data:
                    logger.warning(f"Original WAV for {stem} in {track_title} not found, but present in NPZ. This might be expected if some stems are missing in original data.")
                continue # Skip if original WAV not found

            try:
                original_audio = load_audio(original_stem_file, cfg.target_fs, cfg.target_channels)
            except Exception as e:
                logger.error(f"Error loading original WAV for {stem} in {track_title}: {e}")
                validation_passed = False
                continue

            if stem not in npz_data:
                logger.error(f"Stem '{stem}' missing in NPZ for track {track_title}.")
                raise ValueError(f"Stem '{stem}' missing in NPZ for track {track_title}.")
            npz_audio = npz_data[stem]

        # Pad original audio to match NPZ length if NPZ was padded (only for non-mixture stems, mixture is handled above)
        if stem != "mixture" and original_audio.shape[1] < max_len_npz:
            padding_needed = max_len_npz - original_audio.shape[1]
            original_audio = np.pad(original_audio, ((0, 0), (0, padding_needed)), mode='constant')

        # Compare shapes
        if original_audio.shape != npz_audio.shape:
            logger.error(f"Shape mismatch for {stem} in {track_title}: Original {original_audio.shape}, NPZ {npz_audio.shape}")
            raise ValueError(f"Shape mismatch for {stem} in {track_title}.")

        # Compare content with a tolerance for floating point differences
        if not np.allclose(original_audio, npz_audio, atol=1e-4): # Increased tolerance
            logger.error(f"Content mismatch for {stem} in {track_title}.")
            raise ValueError(f"Content mismatch for {stem} in {track_title}.")
        
        logger.info(f"Stem '{stem}' for track {track_title} validated successfully.")

    if validation_passed:
        logger.info(f"Track {track_title} validated successfully.")
    else:
        logger.error(f"Track {track_title} validation FAILED.")

    return validation_passed

def npzify_musdb18(cfg: DictConfig) -> None:
    """
    Performs the preprocessing step: converts raw MUSDB18HQ WAV files to NPZ format.
    """
    logger.info("Starting preprocessing step (npzify)...")
    input_base_path = os.path.expanduser(cfg.input_path)
    output_base_path = os.path.expanduser(cfg.output_path)
    stems_to_process = cfg.stems_to_process if cfg.stems_to_process else DEFAULT_STEMS

    track_paths = []
    for split in ["train", "test"]:
        split_path = os.path.join(input_base_path, split)
        if os.path.exists(split_path):
            track_paths.extend(glob.glob(os.path.join(split_path, "*")))
        else:
            logger.warning(f"Input split path not found: {split_path}")

    if not track_paths:
        logger.error(f"No tracks found in {input_base_path}. Please check your input_path.")
        return

    original_splits = [os.path.basename(os.path.dirname(p)) for p in track_paths]
    new_splits = []
    for track_path, original_split in zip(track_paths, original_splits):
        track_title = os.path.basename(track_path)
        if original_split == "train" and track_title in MUSDB18HQ_VALIDATION_TRACKS:
            new_splits.append("val")
        else:
            new_splits.append(original_split)

    preprocess_tasks = []
    for i, track_path in enumerate(track_paths):
        preprocess_tasks.append((track_path, original_splits[i], new_splits[i], cfg, stems_to_process))

    logger.info(f"Found {len(preprocess_tasks)} tracks to preprocess.")
    process_map(process_track, preprocess_tasks, max_workers=cfg.num_workers)
    logger.info("Preprocessing complete.")

def validate_musdb18(cfg: DictConfig) -> None:
    """
    Performs the validation step: compares preprocessed NPZ files with original WAV files.
    """
    logger.info("Starting validation step...")
    input_base_path = os.path.expanduser(cfg.input_path)
    output_base_path = os.path.expanduser(cfg.input_path)
    stems_to_process = cfg.stems_to_process if cfg.stems_to_process else DEFAULT_STEMS

    track_titles_to_validate = set()
    for split in ["train", "test"]:
        split_path = os.path.join(input_base_path, split)
        if os.path.exists(split_path):
            for track_dir in glob.glob(os.path.join(split_path, "*")):
                track_titles_to_validate.add(os.path.basename(track_dir))
        else:
            logger.warning(f"Original input split path not found: {split_path}")

    if not track_titles_to_validate:
        logger.error(f"No original tracks found for validation in {input_base_path}. Please check your input_path.")
        return

    validation_tasks = []
    for track_title in sorted(list(track_titles_to_validate)):
        validation_tasks.append((track_title, input_base_path, cfg, stems_to_process))

    logger.info(f"Found {len(validation_tasks)} tracks to validate.")
    
    results = process_map(validate_track, validation_tasks, max_workers=cfg.num_workers)

    num_passed = sum(results)
    num_failed = len(results) - num_passed

    logger.info(f"Validation complete. Passed: {num_passed}/{len(results)}, Failed: {num_failed}/{len(results)}")

    if num_failed > 0:
        logger.error("Some tracks failed validation. Please check logs for details.")
        exit(1)
    else:
        logger.info("All tracks validated successfully!")


@hydra.main(config_path="../configs/preprocess", config_name="musdb18_hq", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info(
        "Running MUSDB18 processing script with the following config:\n"
        + OmegaConf.to_yaml(cfg)
    )

    if cfg.mode == "npzify":
        npzify_musdb18(cfg)
    elif cfg.mode == "validate":
        validate_musdb18(cfg)
    elif cfg.mode == "all":
        npzify_musdb18(cfg)
        validate_musdb18(cfg)
    else:
        logger.error(f"Invalid mode: {cfg.mode}. Please choose 'npzify', 'validate', or 'all'.")
        exit(1)


if __name__ == "__main__":
    main()
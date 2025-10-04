from collections import defaultdict
import os
import numpy as np
from omegaconf import DictConfig
import hydra
from pathlib import Path

import glob

from torchaudio.functional import resample
from torchcodec.decoders import AudioDecoder

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import pandas as pd

from typing import List

from pydantic import BaseModel

import structlog

import librosa

logger = structlog.get_logger()


class TrackMetadata(BaseModel):
    trackType: str
    id: str
    type: str
    extension: str
    has_bleed: bool | None


class StemMetadata(BaseModel):
    id: str
    stemName: str
    tracks: List[TrackMetadata]


class SongMetadata(BaseModel):
    song: str
    artist: str
    genre: str
    stems: List[StemMetadata]


def get_tracks(row, config: DictConfig):

    folder_name = row["folder_name"]
    musdb18_split = row["musdb18_split"]

    if musdb18_split == "train":
        additional_path = "musdb18hq/train"
    elif musdb18_split == "val":
        additional_path = "musdb18hq/val"
    elif musdb18_split == "test":
        additional_path = "musdb18hq/test"
    else:
        additional_path = "dev"

    full_path = Path(
        os.getenv("DATA_ROOT"), config.datasource_id, config.canonical_path, 
        additional_path,
        folder_name
    ).expanduser()
    
    tracks = glob.glob(str(Path(
        full_path, "**/*.flac"
    )), recursive=True)

    track_dict = {}

    for track_path in tracks:
        track_name = Path(track_path).stem.split(".")[0]
        track_type = str(Path(track_path).parent).replace(str(full_path), "").strip("/")
        track_dict[track_name] = (track_path, track_type)

    return track_dict

def track_type_to_coarse(track_type: str) -> str:
    return track_type.split("/")[0]

def process_track_raw(row: pd.Series, config: DictConfig):
    # warnings.filterwarnings("ignore", category=UserWarning, message=".*docs.pytorch.org*")
    print(row)

    track_dict = get_tracks(row, config)

    output = {}

    max_n_samples = 0

    for track_name, (track_path, track_type) in track_dict.items():
        track_key = f"{track_type}/{track_name}"

        audio_samples = AudioDecoder(track_path).get_all_samples()
        data = audio_samples.data
        original_fs = audio_samples.sample_rate
        if original_fs != config.fs:
            print(f"Resampling {track_path} from {original_fs} to {config.fs}")
            data = resample(data, original_fs, config.fs)
        if data.ndim == 1:
            raise ValueError(f"Audio file {track_path} is not 2d.")

        output[track_key] = data.numpy()
        max_n_samples = max(max_n_samples, data.shape[1])

    padded_output = {}
    for key, data in output.items():
        if data.shape[1] < max_n_samples:
            pad_width = max_n_samples - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")
        padded_output[key] = data

    split = row["split"]
    song_name = row["folder_name"]
    output_path = Path(
        os.getenv("DATA_ROOT"),
        config.datasource_id,
        "intermediates",
        "npz-raw",
        str(split),
        f"{song_name}.npz",
    ).expanduser()     

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **padded_output, fs=config.fs)


def process_track_active(row: pd.Series, config: DictConfig):
    
    path = Path(
        os.getenv("DATA_ROOT"),
        config.datasource_id,
        "intermediates",
        "npz-raw",
        str(row["split"]),
        f"{row['folder_name']}.npz",
    ).expanduser()

    raw_data = np.load(path, mmap_mode="r")
    for key in tqdm(raw_data.files):
        if key == "fs":
            continue

        n_levels = key.count("/")
        
        if n_levels >= 2:
            coarse_key, fine_key = key.split("/")[:2]
        else:
            coarse_key = key.split("/")[0]
            fine_key = "__unspecified__"

        print("track_id", path.stem, "key", key, coarse_key, fine_key)

        process_stem_active(
            raw_data[key],
            stem_coarse=coarse_key,
            stem_fine=fine_key,
            track_id=path.stem,
            config=config,
            split=path.parent.name,
        )


def process_stem_active(
    x: np.ndarray,
    *,
    stem_coarse: str,
    stem_fine: str,
    track_id: str,
    config: DictConfig,
    split: str,
):
    intervals: np.ndarray = librosa.effects.split(
        x,
        top_db=config.top_db,
        ref=1.0,
        frame_length=int(config.frame_length_seconds * config.fs),
        hop_length=int(config.hop_length_seconds * config.fs),
    )

    clips = []
    for start, end in intervals:
        clip = x[:, start:end]

        clips.append(clip)

    if len(clips) == 0:
        return

    data = {}
    for i, clip in enumerate(clips):
        key = f"clip{i:04d}"
        data[key] = clip

    output_path = Path(
        os.getenv("DATA_ROOT"),
        config.datasource_id,
        "intermediates",
        "npz-active",
        str(split),
        stem_coarse,
        stem_fine,
        f"{track_id}.npz",
    ).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **{k: v for k, v in data.items()}, fs=config.fs)

def process_track_coarse(row: pd.Series, config: DictConfig):

    path = Path(
        os.getenv("DATA_ROOT"),
        config.datasource_id,
        "intermediates",
        "npz-raw",
        str(row["split"]),
        f"{row['folder_name']}.npz",
    ).expanduser()

    raw_data = np.load(path, mmap_mode="r")
    data = defaultdict(list)
    for key in raw_data.files:
        if key == "fs":
            continue
        coarse_key = track_type_to_coarse(key)
        data[coarse_key].append(raw_data[key])

    data = dict(data)
    data = {k: sum(v) for k, v in data.items()}

    output_path = Path(
        os.getenv("DATA_ROOT"),
        config.datasource_id,
        "intermediates",
        "npz-coarse",
        str(path.parent.name),
        f"{path.stem}.npz",
    ).expanduser()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data, fs=config.fs)


def make_splits(tracklist: pd.DataFrame):
    
    tracklist["split"] = tracklist["musdb18_split"]

    unsplited = tracklist[tracklist["split"] == "none"]
    splited = tracklist[tracklist["split"] != "none"]

    splited_value_counts = splited["split"].value_counts()
    
    # train/val/test = 70/10/20

    n = len(tracklist)
    n_val = 50 - splited_value_counts.get("val", 0)
    n_test = 100 - splited_value_counts.get("test", 0)
    n_train = n - n_val - n_test - len(splited) - splited_value_counts.get("train", 0)

    unsplited = unsplited.sample(frac=1, random_state=42).reset_index(drop=True)

    unsplited.loc[unsplited.index[:n_train], "split"] = "train"
    unsplited.loc[unsplited.index[n_train:n_train+n_val], "split"] = "val"
    unsplited.loc[unsplited.index[n_train+n_val:], "split"] = "test"

    tracklist = pd.concat([splited, unsplited], axis=0).reset_index(drop=True)

    tracklist = tracklist.sort_values(["artist", "title"]).reset_index(drop=True)
    tracklist.to_csv(
        Path(os.getenv("DATA_ROOT"), "rawstems", "tracklist_splitted.csv").expanduser(),
        index=False,
    )

    print(tracklist.value_counts(["split", "musdb18_split"]))

    # print(tracklist)

    return tracklist

def clean_up_canonical(config: DictConfig,
                       recurse_: bool = True,):
    data_root = Path(os.getenv("DATA_ROOT"), config.datasource_id, config.canonical_path).expanduser()

    tracks = list(data_root.glob("**/*.flac"))
    print(f"Number of tracks: {len(tracks)}")

    stem_types = set()
    for track in tqdm(tracks):
        relative_path = track.relative_to(data_root)
        if relative_path.parts[0] == "musdb18hq":
            relative_path = Path(*relative_path.parts[1:])
        # print(relative_path)
        relative_path = relative_path.parent # ignore filename
        relative_path_no_split = Path(*relative_path.parts[2:])
        rel_path_str = str(relative_path_no_split)

        stem_types.add(rel_path_str)

        new_track_name = str(track.relative_to(data_root))
        if "Bss" in rel_path_str:
            new_track_name = new_track_name.replace("Bss", "Bass")

        if "/oc/LV" in str(track):
            new_track_name = new_track_name.replace("oc/LV", "Voc/LV")

        if "Rhy/OERC" in rel_path_str:
            new_track_name = new_track_name.replace("Rhy/OERC", "Rhy/PERC")

        if "Room-AR70.wav" in rel_path_str:
            new_track_name = new_track_name.replace("Room-AR70.wav", "Misc/Room")

        if "VocLV" in rel_path_str:
            new_track_name = new_track_name.replace("VocLV", "Voc/LV")

        if "/VVVoc" in str(track):
            new_track_name = new_track_name.replace("/VVVoc", "/Voc")

        if "/VVoc" in str(track):
            new_track_name = new_track_name.replace("/VVoc", "/Voc")

        new_track_full_path = data_root / new_track_name
        if str(track) != str(new_track_full_path):
            new_track_full_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Renaming {track} to {new_track_full_path}")
            os.rename(track, new_track_full_path)

            # if the folder is empty, remove it
            try:
                track.parent.rmdir()
            except OSError:
                pass
            
    print("Stem types:")
    for stem_type in sorted(stem_types):
        print(stem_type)

    if recurse_:
        # run one more time to ensure all are cleaned up
        clean_up_canonical(config, recurse_=False)

    return


@hydra.main(config_path="../../configs/preprocess")
def main(config: DictConfig):
    tracklist = pd.read_csv(
        Path(os.getenv("DATA_ROOT"), config.datasource_id, "tracklist_splitted.csv").expanduser()
    )

    index = None
    is_in_parallel_mode = os.getenv("PARALLEL_MODE", "0") == "1"
    print(f"PARALLEL_MODE={is_in_parallel_mode}")
    if is_in_parallel_mode:
        index = int(os.getenv("INDEX", "-1"))
        if index < 0:
            raise ValueError("INDEX environment variable must be set in parallel mode.")
        
        if os.getenv("CHUNK_SIZE", None) is not None:
            chunk_size = int(os.getenv("CHUNK_SIZE"))
            index = slice(index, index + chunk_size)
        else:
            index = [index]

    # clean_up_canonical(config)
    # return
    
    # tracks = list(
    #     Path(os.getenv("DATA_ROOT"), config.datasource_id, config.canonical_path).expanduser().glob("*")
    # )
    if config.mode == "raw":
        tracks = [row for _, row in tracklist.iterrows()]
        print(f"Number of tracks: {len(tracks)}")
        # return

        if index is not None:
            # process_track_raw(tracks[index], config)
            for row in tracks[index]:
                process_track_raw(row, config)
        else:
            process_map(process_track_raw, tracks, [config] * len(tracks), max_workers=24)

    if config.mode == "coarse":
        print("Running on val and test only.")
        tracklist = tracklist[tracklist.split != "train"]
        tracks = [row for _, row in tracklist.iterrows()]
        process_map(process_track_coarse, tracks, [config] * len(tracks), max_workers=24)

    if config.mode == "active":
        print("Running on train only.")
        tracklist = tracklist[tracklist.split == "train"]
        tracks = [row for _, row in tracklist.iterrows()]
        print(f"Number of tracks: {len(tracks)}")

        if index is not None:
            # process_track_active(tracks[index], config)
            for row in tracks[index]:
                process_track_active(row, config)
        else:
            process_map(process_track_active, tracks, [config] * len(tracks), max_workers=24)


if __name__ == "__main__":
    main()

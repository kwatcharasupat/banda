import json
import os
import numpy as np
from omegaconf import DictConfig
import hydra
from pathlib import Path


import warnings

import torchaudio as ta
from torchaudio.functional import resample
from torch.nn import functional as F

from tqdm.contrib.concurrent import process_map

import pandas as pd

from typing import List

from pydantic import BaseModel

import structlog

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


def process_track_coarse(path: Path, config: DictConfig):
    # warnings.filterwarnings("ignore", category=UserWarning, message=".*docs.pytorch.org*")

    track_name = path.stem

    splits = pd.read_csv(
        Path(os.getenv("DATA_ROOT"), config.datasource_id, "splits.csv").expanduser()
    ).set_index("song_id")
    fold = splits.loc[track_name, "split"]

    if fold == 4:
        split = "val"
    elif fold == 5:
        split = "test"
    else:
        split = "train"

    data_json_path = Path(path, "data.json")

    with open(data_json_path, "r") as f:
        data_json = json.load(f)
        song_metadata = SongMetadata.model_validate(data_json)

    data = {}

    max_length = 0

    for stem in song_metadata.stems:
        stem_name = stem.stemName
        stem_data = []
        for track in stem.tracks:
            track_path = Path(path, stem_name, f"{track.id}.{track.extension}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, original_fs = ta.load(track_path)
            if original_fs != config.fs:
                x = resample(x, original_fs, config.fs)

            max_length = max(max_length, x.shape[1])
            stem_data.append(x)

        stem_data = [
            F.pad(x, (0, max_length - x.shape[1]), "constant", 0) for x in stem_data
        ]
        data[stem_name] = sum(stem_data)

    # recompute mixture
    if config.recompute_mixture:
        max_n_samples = max(x.shape[1] for x in data.values())
        data = {
            k: F.pad(v, (0, max_n_samples - v.shape[1]), "constant", 0)
            for k, v in data.items()
        }

        data["mixture"] = sum(data.values())

    output_path = Path(
        os.getenv("DATA_ROOT"),
        config.datasource_id,
        "intermediates",
        "npz-coarse",
        str(split),
        f"{track_name}.npz",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **{k: v.cpu().numpy() for k, v in data.items()}, fs=config.fs)


@hydra.main(config_path="../../configs/preprocess", config_name="moisesdb")
def main(config: DictConfig):
    tracks = list(
        Path(os.getenv("DATA_ROOT"), config.datasource_id, "canonical").glob("*/*")
    )

    stems = set(
        list(path.stem for track in tracks for path in track.iterdir() if path.is_dir())
    )
    print(stems)
    exit()

    if config.mode == "coarse":
        process_map(
            process_track_coarse, tracks, [config] * len(tracks), max_workers=16
        )
    else:
        raise NotImplementedError(f"Mode {config.mode} not implemented.")


if __name__ == "__main__":
    main()

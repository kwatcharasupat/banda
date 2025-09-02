

import os
import numpy as np
from omegaconf import DictConfig
import hydra
from pathlib import Path

import structlog
logger = structlog.get_logger()

import torchaudio as ta
from torchaudio.functional import resample
import torch
from torch.nn import functional as F

from tqdm.contrib.concurrent import process_map


def process_track(path: Path, config: DictConfig):
    
    track_name = path.stem
    
    if path.parent.name == "test":
        output_split = "test"
    else:
        if path.name in config.validation_tracks:
            output_split = "val"
        else:
            output_split = "train"

    logger.info("Processing track", track=path.name, split=output_split)
    
    data = {}
    
    for stem_path in path.iterdir():
        
        if config.recompute_mixture and stem_path.stem == "mixture":
            continue
        
        x, original_fs = ta.load(stem_path)

        if original_fs != config.fs:
            x = resample(x, original_fs, config.fs)

        data[stem_path.stem] = x

    # recompute mixture
    if config.recompute_mixture:
        
        max_n_samples = max(x.shape[1] for x in data.values())
        data = {k: F.pad(
            v,
            (0, max_n_samples - v.shape[1]),
            "constant",
            0
        ) for k, v in data.items()}
        
        data['mixture'] = sum(data.values())

    output_path = Path(os.getenv("DATA_ROOT"), config.datasource_id, "intermediates", "npz", output_split, f"{track_name}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        **{k: v.cpu().numpy() for k, v in data.items()},
        fs=config.fs
    )

@hydra.main(config_path="../../configs/preprocess", config_name="musdb18hq")
def main(config: DictConfig):

    tracks = list(Path(os.getenv("DATA_ROOT"), config.datasource_id, "canonical").glob("*/*"))
    
    process_map(
        process_track,
        tracks,
        [config] * len(tracks),
        max_workers=16
    )

if __name__ == "__main__":
    main()
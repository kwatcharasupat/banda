

import os
from typing import Dict, List, Literal

import numpy as np
from pydantic import BaseModel
from banda.data.datasources.base import BaseRegisteredDatasource, DatasourceParams, TrackIdentifier

from pathlib import Path

import structlog
logger = structlog.get_logger(__name__)

MUSDB18Stem = Literal["vocals", "drums", "bass", "other"]

class MUSDB18HQDatasourceParams(DatasourceParams):
    split: str
    load_duration: bool = False
    stems: Dict[str, List[MUSDB18Stem]] = {
        "vocals": ["vocals"],
        "drums": ["drums"],
        "bass": ["bass"],
        "other": ["other"]
    }


class MUSDB18HQDatasource(BaseRegisteredDatasource):
    
    DATASOURCE_ID : str = "musdb18hq"
    
    def __init__(self, *, config: DatasourceParams):
        super().__init__(config=config)
        self.config = MUSDB18HQDatasourceParams.model_validate(config)
        logger.info("Loading tracks for MUSDB18HQ with", config=config)
        self.tracks = self._load_tracks()

    def _load_tracks(self) -> List[TrackIdentifier]:

        datasource_path = Path(os.getenv("DATA_ROOT"), self.DATASOURCE_ID, "intermediates", "npz", self.config.split).expanduser()
        if not datasource_path.exists():
            canonical_path = Path(os.getenv("DATA_ROOT"), self.DATASOURCE_ID, "canonical").expanduser()
            if canonical_path.exists():
                raise RuntimeError(f"Canonical path {canonical_path} exists, but intermediate path {datasource_path} does not. Please run the preprocessing step.")
            else:
                raise RuntimeError(f"Datasource path {datasource_path} does not exist. Please download the MUSDB18HQ dataset and set up the DATA_ROOT environment variable correctly.")

        tracks = [
            TrackIdentifier(
                full_path=path.absolute().as_posix(),
                sources=self.config.stems,
                duration_samples=self._get_duration_samples(path) if self.config.load_duration else None
            )
            for path in datasource_path.iterdir()
        ]
        return tracks

    def _get_duration_samples(self, path: Path) -> int:
        data = np.load(path, mmap_mode='r')
        return data[list(self.config.stems.keys())[0]].shape[1]

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index):
        return self.tracks[index]
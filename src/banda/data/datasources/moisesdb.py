

import os
from typing import Dict, List, Literal

import numpy as np
from pydantic import BaseModel
from banda.data.datasources.base import BaseRegisteredDatasource, DatasourceParams, TrackIdentifier

from pathlib import Path

import structlog
logger = structlog.get_logger(__name__)

MoisesDBStem = Literal['other_keys', 'bass', 'guitar', 'percussion', 'wind', 'bowed_strings', 'drums', 'other_plucked', 'piano', 'other', 'vocals']
MoisesDBStemShortCut = {'_moises_vdbo': {
        "other_keys": ["other_keys"],
        "bass": ["bass"],
        "guitar": ["guitar"],
        "percussion": ["percussion"],
        "wind": ["wind"],
        "bowed_strings": ["bowed_strings"],
        "drums": ["drums"],
        "other_plucked": ["other_plucked"],
        "piano": ["piano"],
        "other": ["other"],
        "vocals": ["vocals"]
    }}



class MoisesDBDatasourceParams(DatasourceParams):
    split: str
    load_duration: bool = False
    stems: Dict[str, List[MoisesDBStem]] | str = MoisesDBStemShortCut['_moises_vdbo']


class MoisesDBDatasource(BaseRegisteredDatasource):
    
    DATASOURCE_ID : str = "moisesdb"
    
    def __init__(self, *, config: DatasourceParams):
        super().__init__(config=config)
        
        if self.config.stems in MoisesDBStemShortCut:
            self.config.stems = MoisesDBStemShortCut[self.config.stems]
        self.config = MoisesDBDatasourceParams.model_validate(config)
        logger.info("Loading tracks for MoisesDB with", config=config)
        self.tracks = self._load_tracks()

    def _load_tracks(self) -> List[TrackIdentifier]:

        datasource_path = Path(os.getenv("DATA_ROOT"), self.DATASOURCE_ID, "intermediates", "npz-coarse", self.config.split).expanduser()
        if not datasource_path.exists():
            canonical_path = Path(os.getenv("DATA_ROOT"), self.DATASOURCE_ID, "canonical").expanduser()
            if canonical_path.exists():
                raise RuntimeError(f"Canonical path {canonical_path} exists, but intermediate path {datasource_path} does not. Please run the preprocessing step.")
            else:
                raise RuntimeError(f"Datasource path {datasource_path} does not exist. Please download the MoisesDB dataset and set up the DATA_ROOT environment variable correctly.")

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
        first_key = [k for k in data.keys() if k in self.config.stems][0]
        return data[first_key].shape[1]

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index):
        return self.tracks[index]
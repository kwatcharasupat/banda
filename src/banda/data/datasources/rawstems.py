import os
from typing import Dict, List, Literal

import numpy as np
from banda.data.datasources.base import (
    BaseRegisteredDatasource,
    BaseStemWiseDatasource,
    DatasourceParams,
    TrackIdentifier,
)

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

RawStemsCoarseStem = Literal['Bass', 'Gtr', 'Kbs', 'Misc', 'Orch', 'Rhy', 'Synth', 'Voc']

RawStemsStemShortCut = {
    "_rs_coarse": {
        "Bass": ["Bass"],
        "Gtr": ["Gtr"],
        "Kbs": ["Kbs"],
        "Misc": ["Misc"],
        "Orch": ["Orch"],
        "Rhy": ["Rhy"],
        "Synth": ["Synth"],
        "Voc": ["Voc"],
    },
    "_rs_full": {
        "Bass": ["__unspecified__", "SB"],
        "Gtr": ["AG", "EG"],
        "Kbs": ["EP", "MTR", "OR", "PN", "SYNTH"],
        "Misc": ["__unspecified__", "Room"],
        "Orch": ["BR", "STR", "WW"],
        "Rhy": ["DK", "PERC"],
        "Synth": ["__unspecified__"],
        "Voc": ["BV", "GRP", "LV"],
    },
}


class RawStemsDatasourceParams(DatasourceParams):
    split: str
    load_duration: bool = False
    stems: Dict[str, List[RawStemsCoarseStem]] | str | None = RawStemsStemShortCut[
        "_rs_coarse"
    ]

    mode: Literal["coarse", "fine", "raw", "active"] = "coarse"


class RawStemsDatasource(BaseRegisteredDatasource):
    DATASOURCE_ID: str = "rawstems"

    def __init__(self, *, config: DatasourceParams):
        super().__init__(config=config)
        self.config = RawStemsDatasourceParams.model_validate(config)

        if (
            isinstance(self.config.stems, str)
            and self.config.stems in RawStemsStemShortCut
        ):
            self.config.stems = RawStemsStemShortCut[self.config.stems]

        logger.info("Loading tracks for RawStems with", config=config)
        self.tracks = self._load_tracks()

    def _get_duration_samples(self, path: Path) -> int:
        data = np.load(path, mmap_mode="r")
        first_key = next(iter(data.keys()))
        return data[first_key].shape[1]

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        return self.tracks[index]

    def _load_tracks(self) -> List[TrackIdentifier]:
        datasource_path = Path(
            os.getenv("DATA_ROOT"),
            self.DATASOURCE_ID,
            "intermediates",
            f"npz-{self.config.mode}",
            self.config.split,
        ).expanduser()
        if not datasource_path.exists():
            canonical_path = Path(
                os.getenv("DATA_ROOT"), self.DATASOURCE_ID, "canonical"
            ).expanduser()
            if canonical_path.exists():
                raise RuntimeError(
                    f"Canonical path {canonical_path} exists, but intermediate path {datasource_path} does not. Please run the preprocessing step."
                )
            else:
                raise RuntimeError(
                    f"Datasource path {datasource_path} does not exist. Please download the RawStems dataset and set up the DATA_ROOT environment variable correctly."
                )

        tracks = [
            TrackIdentifier(
                full_path=path.absolute().as_posix(),
                sources=self.config.stems,
                duration_samples=self._get_duration_samples(path)
                if self.config.load_duration
                else None,
            )
            for path in datasource_path.iterdir()
        ]
        return tracks



class RawStemsStemWiseDatasource(BaseStemWiseDatasource):
    DATASOURCE_ID: str = "rawstems"

    def __init__(self, *, config: DatasourceParams):
        super().__init__(config=config,
                         DatasourceParamsModel=RawStemsDatasourceParams,
                         StemsShortCut=RawStemsStemShortCut,)
        

if __name__ == "__main__":

    datasource = RawStemsStemWiseDatasource(
        config=RawStemsDatasourceParams(
            split="train",
            mode="active",
            stems="_rs_coarse",
        )
    )

    print(f"Number of composite stems: {datasource.n_composite_stems}")

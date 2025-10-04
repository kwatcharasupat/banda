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

MoisesDBCoarseStem = Literal[
    "other_keys",
    "bass",
    "guitar",
    "percussion",
    "wind",
    "bowed_strings",
    "drums",
    "other_plucked",
    "piano",
    "other",
    "vocals",
]

MoisesDBStemShortCut = {
    "_moises_coarse": {
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
        "vocals": ["vocals"],
    },
    "_moises_vdbo": {
        "vocals": ["vocals"],
        "drums": ["drums"],
        "bass": ["bass"],
        "other": [
            "other_keys",
            "guitar",
            "percussion",
            "wind",
            "bowed_strings",
            "other_plucked",
            "piano",
            "other",
        ],
    },
    "_moises_full": {
        "bass": [
            "bass/bass_guitar",
            "bass/bass_synthesizer_(moog_etc)",
            "bass/contrabass",
        ],
        "bowed_strings": [
            "bowed_strings/cello_(solo)",
            "bowed_strings/cello_section",
            "bowed_strings/other_strings",
            "bowed_strings/string_section",
        ],
        "drums": [
            "drums/cymbals",
            "drums/drum_machine",
            "drums/full_acoustic_drumkit",
            "drums/hi_hat",
            "drums/kick_drum",
            "drums/overheads",
            "drums/snare_drum",
            "drums/toms",
        ],
        "guitar": [
            "guitar/acoustic_guitar",
            "guitar/clean_electric_guitar",
            "guitar/distorted_electric_guitar",
        ],
        "other": ["other/fx"],
        "other_keys": [
            "other_keys/organ,_electric_organ",
            "other_keys/other_sounds_(hapischord,_melotron_etc)",
            "other_keys/synth_lead",
            "other_keys/synth_pad",
        ],
        "other_plucked": ["other_plucked/banjo,_mandolin,_ukulele,_harp_etc"],
        "percussion": [
            "percussion/a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)",
            "percussion/pitched_percussion_(mallets,_glockenspiel,_..",
        ],
        "piano": [
            "piano/electric_piano_(rhodes,_wurlitzer,_piano_sound_alike)",
            "piano/grand_piano",
        ],
        "vocals": [
            "vocals/background_vocals",
            "vocals/lead_female_singer",
            "vocals/lead_male_singer",
            "vocals/other_(vocoder,_beatboxing_etc)",
        ],
        "wind": [
            "wind/brass_(trumpet,_trombone,_french_horn,_brass_etc)",
            "wind/flutes_(piccolo,_bamboo_flute,_panpipes,_flutes_etc)",
            "wind/other_wind",
            "wind/reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)",
        ],
    },
}

# make single stem short hands
for stem in MoisesDBCoarseStem.__args__:
    other_stems = [s for s in MoisesDBCoarseStem.__args__ if s != stem]
    MoisesDBStemShortCut[f"_moises_{stem}"] = {stem: [stem], "other": other_stems}


class MoisesDBDatasourceParams(DatasourceParams):
    split: str
    load_duration: bool = False
    stems: Dict[str, List[MoisesDBCoarseStem]] | str | None = MoisesDBStemShortCut[
        "_moises_vdbo"
    ]

    mode: Literal["coarse", "fine", "raw", "active"] = "coarse"


class MoisesDBDatasource(BaseRegisteredDatasource):
    DATASOURCE_ID: str = "moisesdb"

    def __init__(self, *, config: DatasourceParams):
        super().__init__(config=config)
        self.config = MoisesDBDatasourceParams.model_validate(config)

        if (
            isinstance(self.config.stems, str)
            and self.config.stems in MoisesDBStemShortCut
        ):
            self.config.stems = MoisesDBStemShortCut[self.config.stems]

        logger.info("Loading tracks for MoisesDB with", config=config)
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
                    f"Datasource path {datasource_path} does not exist. Please download the MoisesDB dataset and set up the DATA_ROOT environment variable correctly."
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


class MoisesDBStemWiseDatasource(BaseStemWiseDatasource):
    DATASOURCE_ID: str = "moisesdb"

    def __init__(self, *, config: DatasourceParams):
        super().__init__(
            config=config,
            DatasourceParamsModel=MoisesDBDatasourceParams,
            StemsShortCut=MoisesDBStemShortCut,
        )

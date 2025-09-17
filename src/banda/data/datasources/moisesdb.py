from collections import defaultdict
import os
from typing import Dict, List, Literal

import numpy as np
from tqdm import tqdm
from banda.data.datasources.base import (
    BaseRegisteredDatasource,
    DatasourceParams,
    TrackIdentifier,
)
from pydantic import BaseModel

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


class StemIdentifier(BaseModel):
    full_path: str
    name: str
    coarse_stem: str
    fine_stem: str
    n_clips: int


class MoisesDBStemWiseDatasource(BaseRegisteredDatasource):
    DATASOURCE_ID: str = "moisesdb"

    def __init__(self, *, config: DatasourceParams):
        super().__init__(config=config)
        self.config = MoisesDBDatasourceParams.model_validate(config)

        assert self.config.mode in ["active"], (
            "MoisesDBStemWiseDatasource only supports mode='active'"
        )

        if isinstance(self.config.stems, str) and self.config.stems in MoisesDBStemShortCut:
            self.config.stems = MoisesDBStemShortCut[self.config.stems]

        logger.info("Loading tracks for MoisesDB with", config=config)
        self.npz = self._load_npzs()

    def _get_duration_samples(self, path: Path) -> int:
        data = np.load(path, mmap_mode="r")
        first_key = next(iter(data.keys()))
        return data[first_key].shape[1]

    def __len__(self):
        return len(self.npz)

    def __getitem__(self, index):
        return self.tracks[index]

    @property
    def composite_stems(self) -> List[str]:
        return list(self.config.stems.keys())

    @property
    def n_composite_stems(self) -> int:
        return len(self.config.stems)

    @property
    def coarse_stems(self) -> List[str]:
        return list(self.npzs_by_coarse.keys())

    @property
    def n_coarse_stems(self) -> int:
        return len(self.coarse_stems)

    @property
    def fine_stems(self) -> List[str]:
        fine_stems = []
        for coarse_stem, fine_dict in self.npzs_by_fine.items():
            fine_stems.extend(
                [f"{coarse_stem}/{fine_stem}" for fine_stem in fine_dict.keys()]
            )
        return list(fine_stems)

    @property
    def n_fine_stems(self) -> int:
        return len(self.fine_stems)

    @property
    def fine_stems_by_coarse(self) -> Dict[str, List[str]]:
        return {
            coarse_stem: list(fine_dict.keys())
            for coarse_stem, fine_dict in self.npzs_by_fine.items()
        }

    @property
    def n_fine_stems_by_coarse(self) -> Dict[str, int]:
        return {
            coarse_stem: len(fine_dict)
            for coarse_stem, fine_dict in self.npzs_by_fine.items()
        }

    def get_coarse(self, *, coarse_stem: str) -> List[StemIdentifier]:
        return self.npzs_by_coarse.get(coarse_stem, [])

    def get_coarse_length(self, *, coarse_stem: str) -> int:
        if not hasattr(self, "_coarse_length_cache"):
            self._coarse_length_cache = {
                coarse_stem: len(npzs)
                for coarse_stem, npzs in self.npzs_by_coarse.items()
            }

        return self._coarse_length_cache.get(coarse_stem, None)

    def get_fine_length(self, *, coarse_stem: str, fine_stem: str | None) -> int:
        if fine_stem is None:
            return sum(self._fine_length_cache.get(coarse_stem, {}).values())

        if not hasattr(self, "_fine_length_cache"):
            self._fine_length_cache = {
                coarse_stem: {
                    fine_stem: len(npzs) for fine_stem, npzs in fine_dict.items()
                }
                for coarse_stem, fine_dict in self.npzs_by_fine.items()
            }

        return self._fine_length_cache.get(coarse_stem, {}).get(fine_stem, None)

    def get_fine(
        self, *, coarse_stem: str, fine_stem: str | None
    ) -> List[StemIdentifier]:
        if fine_stem is None:
            return self.npzs_by_stem.get(coarse_stem, [])

        return self.npzs_by_fine.get(coarse_stem, {}).get(fine_stem, [])

    @property
    def npzs_by_coarse(self) -> Dict[str, List[StemIdentifier]]:
        if not hasattr(self, "_npz_by_coarse"):
            _npzs_by_coarse = defaultdict(list)
            for npz in self.npz:
                _npzs_by_coarse[npz.coarse_stem].append(npz)
            self._npzs_by_coarse = dict(_npzs_by_coarse)

        return self._npzs_by_coarse

    @property
    def npzs_by_fine(self) -> Dict[str, Dict[str, List[StemIdentifier]]]:
        if not hasattr(self, "_npz_by_fine"):
            _npzs_by_fine = defaultdict(lambda: defaultdict(list))
            for npz in self.npz:
                _npzs_by_fine[npz.coarse_stem][npz.fine_stem].append(npz)
            self._npzs_by_fine = {k: dict(v) for k, v in _npzs_by_fine.items()}

        return self._npzs_by_fine

    @property
    def npzs_by_stem(self) -> Dict[str, List[StemIdentifier]]:
        if not hasattr(self, "_npz_by_stem"):
            _npzs_by_stem = defaultdict(list)

            for composite_stem, constituent_stems in self.config.stems.items():
                for stem in constituent_stems:
                    if "/" not in stem:
                        coarse_stem, fine_stem = stem, None
                    else:
                        coarse_stem, fine_stem = stem.split("/", 1)

                    stem_npzs = self.get_fine(
                        coarse_stem=coarse_stem, fine_stem=fine_stem
                    )
                    _npzs_by_stem[composite_stem].extend(stem_npzs)

            self._npzs_by_stem = dict(_npzs_by_stem)

        return self._npzs_by_stem

    def _load_npzs(self) -> List[TrackIdentifier]:
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

        npzs = []

        for path in tqdm(
            datasource_path.glob("*/*/*.npz"), desc="Loading MoisesDB npzs"
        ):
            track_id = path.stem
            fine_stem = path.parent.stem
            coarse_stem = path.parent.parent.stem

            should_include = self._check_should_include(coarse_stem, fine_stem)
            if not should_include:
                continue

            zip = np.load(path, mmap_mode="r")
            files = list(zip.files)
            files = [f for f in files if f.startswith("clip")]
            n_clips = len(files)

            npz_entry = StemIdentifier(
                full_path=path.absolute().as_posix(),
                name=track_id,
                coarse_stem=coarse_stem,
                fine_stem=fine_stem,
                n_clips=n_clips,
            )

            npzs.append(npz_entry)

        return npzs

    def _check_should_include(self, coarse_stem: str, fine_stem: str) -> bool:
        if self.config.stems is None:
            return True

        for _, constituent_stems in self.config.stems.items():
            if (
                coarse_stem in constituent_stems
                or f"{coarse_stem}/{fine_stem}" in constituent_stems
            ):
                return True

        return False

    def get_coarse_by_index(self, coarse_stem: str, index: int) -> StemIdentifier:
        coarse_length = self.get_coarse_length(coarse_stem=coarse_stem)
        if coarse_length is None:
            raise ValueError(f"Coarse stem '{coarse_stem}' not found in dataset.")

        return self.npzs_by_coarse[coarse_stem][index]

    def get_fine_by_index(
        self, coarse_stem: str, fine_stem: str, index: int
    ) -> StemIdentifier:
        fine_length = self.get_fine_length(coarse_stem=coarse_stem, fine_stem=fine_stem)
        if fine_length is None:
            raise ValueError(
                f"Fine stem '{coarse_stem}/{fine_stem}' not found in dataset."
            )

        return self.npzs_by_fine[coarse_stem][fine_stem][index]

    def resolve_stem_to_composite(
        self, coarse_stem: str, fine_stem: str | None = None
    ) -> str:
        for composite_stem, constituent_stems in self.config.stems.items():
            if coarse_stem in constituent_stems:
                return composite_stem

            if (
                fine_stem is not None
                and f"{coarse_stem}/{fine_stem}" in constituent_stems
            ):
                return composite_stem

        raise ValueError(
            f"Stem '{coarse_stem}' or '{coarse_stem}/{fine_stem}' not found in any composite stem."
        )

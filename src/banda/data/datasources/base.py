from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
from typing_extensions import Type
from pydantic import BaseModel
from torch.utils.data import Dataset

from pathlib import Path
import numpy as np
import os

from banda.utils import WithClassConfig, BaseConfig
from structlog import get_logger

logger = get_logger(__name__)


class DatasourceParams(BaseConfig):
    pass


class DatasourceConfig(WithClassConfig[DatasourceParams]):
    pass


class TrackIdentifier(BaseModel):
    full_path: str
    duration_samples: int | None = None
    sources: Dict[str, List[str]] | None = None


class StemIdentifier(BaseModel):
    full_path: str
    name: str
    coarse_stem: str
    fine_stem: str
    n_clips: int


class DatasourceRegistry(type):
    # from https://charlesreid1.github.io/python-patterns-the-registry.html

    DATASOURCE_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.DATASOURCE_REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.DATASOURCE_REGISTRY)


class BaseRegisteredDatasource(Dataset, metaclass=DatasourceRegistry):
    def __init__(self, *, config: DatasourceParams):
        super().__init__()
        self.config = config




class BaseStemWiseDatasource(BaseRegisteredDatasource):
    DATASOURCE_ID: str

    def __init__(self, *, config: DatasourceParams,
                 DatasourceParamsModel: Type[DatasourceParams] = None,
                 StemsShortCut: Dict[str, List[str]] = None,
                 ):
        super().__init__(config=config)
        self.config = DatasourceParamsModel.model_validate(config)

        assert self.config.mode in ["active"], (
            "BaseStemWiseDatasource only supports mode='active'"
        )

        if (
            isinstance(self.config.stems, str)
            and self.config.stems in StemsShortCut
        ):
            self.config.stems = StemsShortCut[self.config.stems]

        logger.info("Loading tracks for RawStems with", config=config)
        self.npz = self._load_npzs()

    def _get_duration_samples(self, path: Path) -> int:
        data = np.load(path, mmap_mode="r")
        first_key = next(iter(data.keys()))
        return data[first_key].shape[1]

    def __len__(self):
        return len(self.npz)

    def __getitem__(self, index):
        logger.error("BaseStemWiseDatasource does not implement __getitem__")
        raise ValueError("BaseStemWiseDatasource does not implement __getitem__")

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
                    f"Datasource path {datasource_path} does not exist. Please download the RawStems dataset and set up the DATA_ROOT environment variable correctly."
                )

        npzs = []

        for path in tqdm(
            datasource_path.glob("*/*/*.npz"), desc="Loading RawStems npzs"
        ):
            track_id = path.stem
            fine_stem = path.parent.stem
            coarse_stem = path.parent.parent.stem

            # print("track_id", track_id, coarse_stem, fine_stem)

            should_include = self._check_should_include(coarse_stem, fine_stem)
            if not should_include:
                print(
                    f"Skipping track {track_id} with stem {coarse_stem}/{fine_stem} as it's not in the configured stems."
                )
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
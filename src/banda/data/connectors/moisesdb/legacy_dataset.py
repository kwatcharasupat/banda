import os
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from bandx.data.datasets.base import SourceSeparationDataset
from bandx.typing import input_dict


DBFS_HOP_SIZE = int(0.125 * 44100)
DBFS_CHUNK_SIZE = int(1 * 44100)


class MoisesDBBaseDataset(SourceSeparationDataset, ABC):
    __DEFAULT_DATA_PATH__: str = os.path.expandvars(
        os.path.join("$DATA_ROOT", "moisesdb")
    )
    __DEFAULT_FS__: int = 44100
    __DEFAULT_TRAIN_FOLDS__: List[int] = [1, 2, 3]
    __DEFAULT_VAL_FOLDS__: List[int] = [4]
    __DEFAULT_TEST_FOLDS__: List[int] = [5]
    __DEFAULT_QUERY_FILENAME__: str = "query"
    __DEFAULT_SPLIT_FILENAME__: str = "splits.csv"
    __DEFAULT_STEM_FILENAME__: str = "stems.csv"

    def __init__(
        self,
        *,
        split: str,
        data_path: str = __DEFAULT_DATA_PATH__,
        fs: int = __DEFAULT_FS__,
        return_stems: Union[bool, List[str]] = False,
        npy_memmap: bool = True,
        recompute_mixture: bool = False,
        train_folds: Optional[List[int]] = None,
        val_folds: Optional[List[int]] = None,
        test_folds: Optional[List[int]] = None,
        query_filename: str = __DEFAULT_QUERY_FILENAME__,
        split_filename: str = __DEFAULT_SPLIT_FILENAME__,
        stem_filename: str = __DEFAULT_STEM_FILENAME__,
    ) -> None:
        if train_folds is None:
            train_folds = self.__DEFAULT_TRAIN_FOLDS__
        if val_folds is None:
            val_folds = self.__DEFAULT_VAL_FOLDS__
        if test_folds is None:
            test_folds = self.__DEFAULT_TEST_FOLDS__

        split_path = os.path.join(data_path, "splits.csv")
        splits = pd.read_csv(split_path)

        metadata_path = os.path.join(data_path, "stems.csv")
        metadata = pd.read_csv(metadata_path)

        if split == "train":
            folds = train_folds
        elif split == "val":
            folds = val_folds
        elif split == "test":
            folds = test_folds
        else:
            raise NameError

        files = splits[splits["split"].isin(folds)]["song_id"].tolist()
        metadata = metadata[metadata["song_id"].isin(files)]

        super().__init__(
            split=split,
            stems=["mixture"],
            files=files,
            data_path=data_path,
            fs=fs,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
        )

        self.folds = folds

        self.metadata = metadata.rename(
            columns={k: k.replace(" ", "_") for k in metadata.columns}
        )

        self.song_to_stem = (
            metadata.set_index("song_id")
            .apply(lambda row: row[row == 1].index.tolist(), axis=1)
            .to_dict()
        )
        self.stem_to_song = (
            metadata.set_index("song_id")
            .transpose()
            .apply(lambda row: row[row == 1].index.tolist(), axis=1)
            .to_dict()
        )

        self.true_length = len(self.files)
        self.n_channels = 2

        self.audio_path = os.path.join(data_path, "npy2")

        self.return_stems = return_stems

        self.query_file = query_filename

    def get_full_stem(self, *, stem: str, identifier) -> torch.Tensor:
        song_id = identifier["song_id"]
        # print(song_id)
        path = os.path.join(self.data_path, "npy2", song_id)
        # noinspection PyUnresolvedReferences

        assert self.npy_memmap

        if os.path.exists(os.path.join(path, f"{stem}.npy")):
            audio = np.load(os.path.join(path, f"{stem}.npy"), mmap_mode="r")
        else:
            raise FileNotFoundError(f"{song_id, stem}")

        return audio

    def get_query_stem(self, *, stem: str, identifier) -> torch.Tensor:
        song_id = identifier["song_id"]
        path = os.path.join(self.data_path, "npyq", song_id)
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            audio = np.load(
                os.path.join(path, f"{stem}.{self.query_file}.npy"), mmap_mode="r"
            )
        else:
            raise NotImplementedError

        return audio

    def get_stem(self, *, stem: str, identifier) -> torch.Tensor:
        audio = self.get_full_stem(stem=stem, identifier=identifier)
        return audio

    def get_identifier(self, index):
        return dict(song_id=self.files[index % self.true_length])

    def __getitem__(self, index: int):
        identifier = self.get_identifier(index)
        audio = self.get_audio(identifier)

        mixture = audio["mixture"].copy()

        if isinstance(self.return_stems, list):
            sources = {
                stem: audio.get(stem, np.zeros_like(mixture))
                for stem in self.return_stems
            }
        elif isinstance(self.return_stems, bool):
            if self.return_stems:
                sources = {
                    stem: audio[stem].copy()
                    for stem in self.song_to_stem[identifier["song_id"]]
                }
            else:
                sources = None
        else:
            raise ValueError

        return input_dict(
            mixture=mixture,
            sources=sources,
            metadata=identifier,
            modality="audio",
        )


class MoisesDBFullTrackDataset(MoisesDBBaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        return_stems: Union[bool, List[str]] = False,
        npy_memmap: bool = True,
        recompute_mixture: bool = False,
        query_file: str = "query",
    ) -> None:
        super().__init__(
            split=split,
            data_path=data_root,
            return_stems=return_stems,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
            query_filename=query_file,
        )

    def __len__(self) -> int:
        return self.true_length

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}
        for stem in stems:
            audio[stem] = self.get_stem(stem=stem, identifier=identifier)

        return audio

    def get_audio(self, identifier: Dict[str, Any]):
        if self.recompute_mixture:
            audio = self._get_audio(
                self.song_to_stem[identifier["song_id"]], identifier=identifier
            )
            audio["mixture"] = self.compute_mixture(audio)
            return audio
        else:
            return self._get_audio(
                self.song_to_stem[identifier["song_id"]] + ["mixture"],
                identifier=identifier,
            )

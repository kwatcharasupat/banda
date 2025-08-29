import math
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional
from ....typing.batch import input_dict
from bandx.data.moisesdb.datasets.query import (
    MoisesDBDeterministicChunkDeterministicQueryDataset,
    MoisesDBFullTrackTestQueryDataset,
    MoisesDBRandomChunkRandomQueryDataset,
)

from bandx.data.moisesdb._taxonomy import moisesdb_taxonomy


import numpy as np


import os
import random


class MoisesDBRandomChunkPrecomputedQueryDataset(MoisesDBRandomChunkRandomQueryDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4.0,
        query_size_seconds: float = 1.0,
        round_query: bool = False,
        min_query_dbfs: float = -40.0,
        min_target_dbfs: float = -36.0,
        min_target_dbfs_step: float = -12.0,
        max_dbfs_tries: int = 10,
        top_k_instrument: int = 10,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap: bool = True,
        allowed_stems=None,
        query_file: str = "query-10s.passt",
        augment=None,
    ) -> None:
        super().__init__(
            data_root,
            split,
            target_length,
            chunk_size_seconds,
            query_size_seconds,
            round_query,
            min_query_dbfs,
            min_target_dbfs,
            min_target_dbfs_step,
            max_dbfs_tries,
            top_k_instrument,
            mixture_stem,
            use_own_query,
            npy_memmap,
            allowed_stems,
            query_file,
            augment,
        )

        # self.allowed_stems = set(self.allowed_stems) & ALLOWED_PRECOMPUTED_QUERY_STEMS
        self.allowed_stems = set(self.allowed_stems)

    def get_query_embedding(self, stem):
        path = os.path.join(self.data_path, "passt-avg")
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            query = np.load(
                os.path.join(path, f"{stem}.{self.query_file}.passt.npy"), mmap_mode="r"
            )
        else:
            raise NotImplementedError

        return query

    def __getitem__(self, index: int):
        mix_identifier = self.get_identifier(index)
        mix_stems = self.song_to_stem[mix_identifier["song_id"]]

        if self.use_own_query:
            possible_stem = mix_stems
        else:
            possible_stem = list(set(mix_stems) & set(self.allowed_stems))

        stem = random.choice(possible_stem)

        zero_target = stem not in mix_stems

        if zero_target:
            audio = self._get_audio(
                [self.mixture_stem],
                identifier=mix_identifier,
                check_dbfs=False,
                no_target=True,
            )
            mixture = audio[self.mixture_stem].copy()
            sources = {"target": np.zeros_like(mixture)}
        else:
            audio = self._get_audio(
                [stem, self.mixture_stem], identifier=mix_identifier, check_dbfs=True
            )
            mixture = audio[self.mixture_stem].copy()
            sources = {"target": audio[stem].copy()}

        query = self.get_query_embedding(stem=stem)

        assert mixture.shape[-1] == self.chunk_size_samples
        # assert query.shape[-1] == self.query_size_samples
        assert sources["target"].shape[-1] == self.chunk_size_samples

        return input_dict(
            mixture=mixture,
            sources=sources,
            query={
                "passt": query,
            },
            metadata={
                "mix": mix_identifier,
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBDeterministicChunkPrecomputedQueryDataset(
    MoisesDBDeterministicChunkDeterministicQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 4.0,
        hop_size_seconds: float = 8.0,
        query_size_seconds: float = 1.0,
        min_query_dbfs: float = -40.0,
        top_k_instrument: int = 10,
        n_queries_per_chunk: int = 1,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap: bool = True,
        allowed_stems: Optional[List[str]] = None,
        query_file: str = "query-10s.passt",
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            query_size_seconds=query_size_seconds,
            min_query_dbfs=min_query_dbfs,
            top_k_instrument=top_k_instrument,
            n_queries_per_chunk=n_queries_per_chunk,
            mixture_stem=mixture_stem,
            use_own_query=use_own_query,
            npy_memmap=npy_memmap,
            allowed_stems=allowed_stems,
            query_file=query_file,
        )

        # self.allowed_stems = set(self.allowed_stems) & ALLOWED_PRECOMPUTED_QUERY_STEMS
        self.allowed_stems = set(self.allowed_stems)

    def get_query_embedding(self, stem):
        path = os.path.join(self.data_path, "passt-avg")
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            query = np.load(
                os.path.join(path, f"{stem}.{self.query_file}.passt.npy"), mmap_mode="r"
            )
        else:
            raise NotImplementedError

        return query

    def index_to_identifiers(self, index: int) -> Tuple[str, str, str, int]:
        row = self.query_tuples.iloc[index]
        mix_id = row["mix"]

        stem = row["stem"]
        mix_chunk_start = row["mix_chunk_start"]

        return mix_id, stem, mix_chunk_start

    def __getitem__(self, index: int):
        mix_id, stem, mix_chunk_start = self.index_to_identifiers(index)

        mix_identifier = dict(song_id=mix_id, chunk_start=mix_chunk_start)

        audio = self._get_audio([stem, self.mixture_stem], identifier=mix_identifier)
        query = self.get_query_embedding(stem=stem)

        mixture = audio[self.mixture_stem].copy()
        sources = {"target": audio[stem].copy()}
        # query = query.copy()

        assert mixture.shape[-1] == self.chunk_size_samples
        # print(query.shape[-1], self.query_size_samples)
        # assert query.shape[-1] == self.query_size_samples
        assert sources["target"].shape[-1] == self.chunk_size_samples

        return input_dict(
            mixture=mixture,
            sources=sources,
            query={
                "passt": query,
            },
            metadata={
                "mix": mix_identifier,
                # "query": query_identifier,
                "stem": stem,
            },
            # modality="audio",
        )

    def generate_index(self):
        durations = pd.read_csv(os.path.join(self.data_path, "durations.csv"))
        durations = (
            durations[["song_id", "duration"]]
            .set_index("song_id")["duration"]
            .to_dict()
        )

        tuples = []

        defaultdict(list)

        for i, song_id in tqdm(enumerate(self.files), total=len(self.files)):
            song_duration = durations[song_id]
            mix_stems = self.song_to_stem[song_id]

            n_mix_chunks = math.floor(
                (song_duration - self.chunk_size_seconds) / self.chunk_hop_size_seconds
            )

            for stem in mix_stems:
                for k in tqdm(range(n_mix_chunks), desc=f"song{i + 1}/{stem}"):
                    mix_chunk_start = int(k * self.chunk_hop_size_samples)

                    tuples.append(
                        dict(
                            mix=song_id,
                            stem=stem,
                            mix_chunk_start=mix_chunk_start,
                        )
                    )

        tuples = pd.DataFrame(tuples)
        return tuples

    def find_query_tuples_or_generate(self):
        return self.generate_index()


class MoisesDBRandomChunkPrecomputedRandomSingleQueryDataset(
    MoisesDBRandomChunkPrecomputedQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4.0,
        query_size_seconds: float = 1.0,
        round_query: bool = False,
        min_query_dbfs: float = -40.0,
        min_target_dbfs: float = -36.0,
        min_target_dbfs_step: float = -12.0,
        max_dbfs_tries: int = 10,
        mixture_stem: str = "mixture",
        npy_memmap: bool = True,
        augment=None,
        allowed_stems=None,
        **kwargs,
    ) -> None:
        # if allowed_stems is None:
        #     allowed_stems = FINE_LEVEL_INSTRUMENTS

        super().__init__(
            data_root=data_root,
            split=split,
            target_length=target_length,
            chunk_size_seconds=chunk_size_seconds,
            query_size_seconds=query_size_seconds,
            round_query=round_query,
            min_query_dbfs=min_query_dbfs,
            min_target_dbfs=min_target_dbfs,
            min_target_dbfs_step=min_target_dbfs_step,
            max_dbfs_tries=max_dbfs_tries,
            top_k_instrument=None,
            mixture_stem=mixture_stem,
            use_own_query=None,
            npy_memmap=npy_memmap,
            allowed_stems=allowed_stems,
            query_file=None,
            augment=augment,
        )

        self.stem_to_song = {
            k: list(v) for k, v in self.stem_to_song.items() if len(v) >= 2
        }

        self.allowed_stems = set(self.allowed_stems) & set(self.stem_to_song.keys())
        self.allowed_stems = {
            s
            for s in self.allowed_stems
            if s in moisesdb_taxonomy.coarse_level_instruments
            or len(
                set(
                    moisesdb_taxonomy.coarse_to_fine[
                        moisesdb_taxonomy.fine_to_coarse[s]
                    ]
                )
                & self.allowed_stems
            )
            > 1
        }

        self.song_to_stem = {
            k: list(set(v) & set(self.allowed_stems))
            for k, v in self.song_to_stem.items()
        }

        if split == "train":
            assert augment

    def __len__(self) -> int:
        return self.target_length

    def get_query_embedding(self, stem, song_id):
        path = os.path.join(self.data_path, "passt-full")
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            query = np.load(os.path.join(path, song_id, f"{stem}.passt.stats.npy"))
            # assert query.shape == (2, 768)
        else:
            raise NotImplementedError

        return query

    def get_nontarget_stems(self, song_id, target_stem):
        song_stems = self.song_to_stem[song_id]

        song_fine_stems = self.song_to_fine_stems[song_id]

        if target_stem in moisesdb_taxonomy.coarse_level_instruments:
            song_coarse_stems = self.song_to_coarse_stems[song_id]

            song_stems = []

            for coarse_stem in song_coarse_stems:
                if coarse_stem == target_stem:
                    continue

                fine_in_coarse = moisesdb_taxonomy.coarse_to_fine[coarse_stem] & set(
                    song_fine_stems
                )
                if len(fine_in_coarse) > 0:
                    song_stems += list(fine_in_coarse)

        else:
            song_stems = [s for s in song_fine_stems if s != target_stem]

        return song_stems

    def __getitem__(self, index: int):
        # print(index, end=" ")

        song1_id = self.get_identifier(index)["song_id"]

        song1_stems = self.song_to_stem[song1_id]
        stem1 = random.choice(song1_stems)

        song1_stems = self.get_nontarget_stems(song1_id, stem1)

        song1_audio = self._get_audio(
            stem1,
            song1_stems,
            {"song_id": song1_id, "index": index, "stem_order": 1},
            target_name="target1",
        )

        audio = song1_audio

        if self.augment is not None:
            audio = self._augment(audio, ["target1"])
            mixture = audio.pop("mixture")
            sources = audio
        else:
            mixture = sum(audio.values())
            sources = {"target1": audio["target1"]}

        query1 = self.get_query_embedding(stem=stem1, song_id=song1_id)

        return input_dict(
            mixture=mixture,
            sources=sources,
            query={"passt": query1},
            metadata={
                "song1_id": song1_id,
                "stem1": stem1,
            },
        )

    def _augment(self, audio, target_stems):
        stack_audio = np.stack(list(audio.values()), axis=0)
        aug_audio = self.augment(torch.from_numpy(stack_audio))
        mixture = torch.sum(aug_audio, axis=0).numpy()

        out = {
            "mixture": mixture,
        }

        for target_stem in target_stems:
            target_idx = list(audio.keys()).index(target_stem)
            out[target_stem] = aug_audio[target_idx].numpy()

        return out

    def _get_audio(  # type: ignore
        self,
        target_stem,
        other_stems,
        identifier: Dict[str, Any],
        check_dbfs: bool = True,
        target_name: str = "target",
    ):
        audio = {}
        for stem in other_stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        audio[target_name] = self.get_full_stem(stem=target_stem, identifier=identifier)

        audio_ = {k: v.copy() for k, v in audio.items()}

        if check_dbfs:
            assert target_stem is not None
            audio = self._chunk_and_check_dbfs(audio_, target_name)
        else:
            first_key = next(iter(audio_.keys()))
            start, end = self._get_start_end(audio_[first_key])
            audio = self._chunk_audio(audio_, start, end)

        return audio


class MoisesDBFullTrackPrecomputedQueryDataset(MoisesDBFullTrackTestQueryDataset):
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        top_k_instrument: int = 10,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap: bool = True,
        allowed_stems: Optional[List[str]] = None,
        query_file: str = "query-10s",
    ) -> None:
        super().__init__(
            data_root,
            split,
            top_k_instrument,
            mixture_stem,
            use_own_query,
            npy_memmap,
            allowed_stems,
            query_file,
        )

        # self.allowed_stems = set(self.allowed_stems) & ALLOWED_PRECOMPUTED_QUERY_STEMS
        self.allowed_stems = set(self.allowed_stems)

    def get_query_embedding(self, stem):
        path = os.path.join(self.data_path, "passt-avg")
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            query = np.load(
                os.path.join(path, f"{stem}.{self.query_file}.passt.npy"), mmap_mode="r"
            )
        else:
            raise NotImplementedError

        return query

    def __getitem__(self, index: int):
        mix_id, query_id, stem = self.index_to_identifiers(index)

        mix_identifier = dict(song_id=mix_id)

        dict(song_id=query_id)

        audio = self._get_audio([stem, "mixture"], identifier=mix_identifier)
        query = self.get_query_embedding(stem=stem)

        mixture = audio["mixture"].copy()
        sources = {stem: audio[stem].copy()}
        query = query.copy()

        return input_dict(
            mixture=mixture,
            sources=sources,
            query={
                "passt": query,
            },
            metadata={
                "mix": mix_identifier,
                "stem": stem,
            },
            # modality="audio",
        )


class MoisesDBDeterministicChunkPrecomputedDeterministicSingleQueryDataset(
    MoisesDBRandomChunkPrecomputedRandomSingleQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4,
        query_size_seconds: float = 1,
        round_query: bool = False,
        min_query_dbfs: float = -40,
        min_target_dbfs: float = -36,
        min_target_dbfs_step: float = -12,
        max_dbfs_tries: int = 10,
        mixture_stem: str = "mixture",
        npy_memmap: bool = True,
        augment=None,
        allowed_stems=None,
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            target_length=target_length,
            chunk_size_seconds=chunk_size_seconds,
            query_size_seconds=query_size_seconds,
            round_query=round_query,
            min_query_dbfs=min_query_dbfs,
            min_target_dbfs=min_target_dbfs,
            min_target_dbfs_step=min_target_dbfs_step,
            max_dbfs_tries=max_dbfs_tries,
            mixture_stem=mixture_stem,
            npy_memmap=npy_memmap,
            augment=augment,
            allowed_stems=allowed_stems,
        )

        index = []
        file_combi_per_stem_combi = max(
            1, self.target_length // len(self.allowed_stems)
        )

        used_files1: Set[str] = set()

        counter_by_stem1: defaultdict[Any, int] = defaultdict(lambda: 0)

        for stem1 in self.allowed_stems:
            file1s = self.stem_to_song[stem1]

            n_combis = min(len(file1s), file_combi_per_stem_combi)

            for _ in range(n_combis):
                file1s = self.stem_to_song[stem1]

                unused_file1s = sorted(set(file1s) - used_files1)
                if len(unused_file1s) > 0:
                    file1 = unused_file1s[0]
                else:
                    file1 = file1s[counter_by_stem1[stem1] % len(file1s)]
                    counter_by_stem1[stem1] += 1

                used_files1.add(file1)
                if len(used_files1) == len(self.files):
                    used_files1.clear()

                index.append(
                    {
                        "song1_id": file1,
                        "stem1": stem1,
                    }
                )

        self.index = index
        self.pair_start_time: Dict = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        # print(index, end=" ")

        identifiers = self.index[index]

        song1_id = identifiers["song1_id"]
        stem1 = identifiers["stem1"]

        if stem1 in moisesdb_taxonomy.coarse_level_instruments:
            song1_coarse_stems = self.song_to_coarse_stems[song1_id]

            song1_stems = []

            for coarse_stem in song1_coarse_stems:
                if coarse_stem == stem1:
                    continue

                fine_in_coarse = set(
                    moisesdb_taxonomy.coarse_to_fine_set[coarse_stem]
                ) & set(self.song_to_fine_stems[song1_id])
                if len(fine_in_coarse) > 0:
                    song1_stems += list(fine_in_coarse)

        else:
            song1_stems = [s for s in self.song_to_fine_stems[song1_id] if s != stem1]

        song1_audio = self._get_audio(
            stem1, song1_stems, {"song_id": song1_id, "index": index, "stem_order": 1}
        )

        target1 = song1_audio.pop("target")

        audio = {**song1_audio, "target1": target1}

        if self.augment is not None:
            audio = self._augment(audio, ["target1"])
            mixture = audio.pop("mixture")
            sources = audio
        else:
            mixture = sum(audio.values())
            sources = {"target1": target1}

        query1 = self.get_query_embedding(stem=stem1, song_id=song1_id)

        # print('got')
        return input_dict(
            mixture=mixture,
            sources=sources,
            query={"passt": query1},
            metadata={
                "song1_id": song1_id,
                "stem1": stem1,
            },
            modality="audio",
        )

    def _get_audio(
        self,
        target_stem,
        other_stems,
        identifier: Dict[str, Any],
        check_dbfs: bool = True,
    ):
        audio = {}
        for stem in other_stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        audio["target"] = self.get_full_stem(stem=target_stem, identifier=identifier)

        audio_ = {k: v.copy() for k, v in audio.items()}

        pair_key = (identifier["index"], identifier["stem_order"])
        start = self.pair_start_time.get(pair_key)

        if start is None:
            first_key = next(iter(audio_.keys()))
            start, end = self._get_start_end(audio_[first_key])
            self.pair_start_time[pair_key] = start
        else:
            end = start + self.chunk_size_samples

        audio = self._chunk_audio(audio_, start, end)

        for v in audio.values():
            assert v.shape[-1] == self.chunk_size_samples

        return audio

    def _get_start_end(self, audio):
        audio.shape[-1]

        frames = librosa.util.frame(
            audio,
            frame_length=self.chunk_size_samples,
            hop_length=self.chunk_size_samples // 2,
        )
        # print(frames.shape)
        rms = np.mean(np.square(frames), axis=(0, 1))  # skip sqrt

        # print(rms.shape)
        max_rms = np.argmax(rms)

        start = max_rms * self.chunk_size_samples // 2

        end = start + self.chunk_size_samples
        return start, end

import math
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from bandx.data.augmentation import Audiomentations
from bandx.data.moisesdb.legacy_dataset import (
    ALL_LEVEL_INSTRUMENTS,
    COARSE_LEVEL_INSTRUMENTS,
    FINE_LEVEL_INSTRUMENTS,
    FINE_TO_COARSE,
    INST_BY_OCCURRENCE,
    MoisesDBFullTrackDataset,
)
from bandx.types import dual_input_dict, input_dict


import numpy as np
import torch


import random
from typing import Any, Dict, List, Tuple, Union, Optional


class MoisesDBBaseQueryDataset(MoisesDBFullTrackDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        return_stems: Union[bool, List[str]] = False,
        npy_memmap: bool = True,
        recompute_mixture: bool = False,
        query_file: str = "query",
        mixture_stem: str = "mixture",
        min_query_dbfs=-40.0,
        min_target_dbfs=None,
        min_target_dbfs_step=None,
        max_dbfs_tries: int = 1,
        top_k_instrument: int = 10,
        allowed_stems=None,
        use_own_query: bool = True,
        augment=None,
        allow_pad: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            data_path=data_root,
            return_stems=return_stems,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
            query_file=query_file,
        )

        self.mixture_stem = mixture_stem

        self.min_query_dbfs = min_query_dbfs

        if min_target_dbfs is None:
            min_target_dbfs = -np.inf
            min_target_dbfs_step = None
            max_dbfs_tries = 1

        self.min_target_dbfs = min_target_dbfs
        self.min_target_dbfs_step = min_target_dbfs_step
        self.max_dbfs_tries = max_dbfs_tries

        self.top_k_instrument = top_k_instrument

        if allowed_stems is None:
            allowed_stems = INST_BY_OCCURRENCE[: self.top_k_instrument]
        else:
            self.top_k_instrument = None

        self.allowed_stems = allowed_stems

        self.song_to_all_stems = {
            k: list(set(v) & set(ALL_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }

        self.song_to_fine_stems = {
            k: list(set(v) & set(FINE_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }

        self.song_to_coarse_stems = {
            k: list(set(v) & set(COARSE_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }

        self.song_to_stem = {
            k: list(set(v) & set(self.allowed_stems))
            for k, v in self.song_to_stem.items()
        }
        self.stem_to_song = {
            k: list(set(v) & set(self.files)) for k, v in self.stem_to_song.items()
        }

        self.queriable_songs = [k for k, v in self.song_to_stem.items() if len(v) > 0]

        self.use_own_query = use_own_query

        if self.use_own_query:
            self.files = [k for k in self.files if len(self.song_to_stem[k]) > 0]
            self.true_length = len(self.files)

        if augment is not None:
            assert self.recompute_mixture
            self.augment = Audiomentations(augment, self.fs)
        else:
            self.augment = None

        self.allow_pad = allow_pad

    def _augment(self, audio, target_stem):
        stack_audio = np.stack(list(audio.values()), axis=0)
        aug_audio = self.augment(torch.from_numpy(stack_audio)).numpy()
        mixture = np.sum(aug_audio, axis=0)

        out = {
            "mixture": mixture,
        }

        if target_stem is not None:
            target_idx = list(audio.keys()).index(target_stem)
            out[target_stem] = aug_audio[target_idx]

        return out

    def _choose_stems_for_augment(self, identifier, target_stem):
        stems_for_song = set(self.song_to_all_stems[identifier["song_id"]])

        stems_ = []
        coarse_level_accounted = set()

        is_none_target = target_stem is None
        is_coarse_target = target_stem in COARSE_LEVEL_INSTRUMENTS

        if is_coarse_target or is_none_target:
            coarse_target = target_stem
        else:
            coarse_target = FINE_TO_COARSE[target_stem]

        fine_level_stems = stems_for_song & FINE_LEVEL_INSTRUMENTS
        coarse_level_stems = stems_for_song & COARSE_LEVEL_INSTRUMENTS

        for s in fine_level_stems:
            coarse_level = FINE_TO_COARSE[s]

            if is_coarse_target and coarse_level == coarse_target:
                continue
            else:
                stems_.append(s)

            coarse_level_accounted.add(coarse_level)

        stems_ += list(coarse_level_stems - coarse_level_accounted)

        if target_stem is not None:
            assert target_stem in stems_, f"stems: {stems_}, target stem: {target_stem}"

            if len(stems_for_song) > 1:
                assert len(stems_) > 1, (
                    f"stems: {stems_}, stems in song: {stems_for_song},\n target stem: {target_stem}"
                )

        assert "mixture" not in stems_

        return stems_


class MoisesDBBaseChunkedQueryDataset(MoisesDBBaseQueryDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 6.0,
        query_size_seconds: float = 10.0,
        min_dbfs: float = -40.0,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        allowed_stems=None,
        npy_memmap: bool = True,
        fs: int = 44100,
    ) -> None:
        super().__init__(
            data_root,
            split,
            return_stems=False,
            npy_memmap=npy_memmap,
            recompute_mixture=True,
            mixture_stem=mixture_stem,
            min_query_dbfs=min_dbfs,
            allowed_stems=allowed_stems,
            use_own_query=use_own_query,
        )

        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = int(chunk_size_seconds * fs)

        self.query_size_seconds = query_size_seconds
        self.query_size_samples = int(query_size_seconds * fs)

    def _get_start_end(self, audio):
        raise NotImplementedError

    def _get_random_start_end(self, audio):
        n_samples = audio.shape[-1]
        start = np.random.randint(0, n_samples - self.chunk_size_samples)
        end = start + self.chunk_size_samples

        return start, end

    def _chunk_audio(self, audio, start, end):
        n_samples = end - start

        audio = {k: v[..., start:end] for k, v in audio.items()}

        for k, v in audio.items():
            if v.shape[-1] < n_samples:
                if self.allow_pad:
                    audio[k] = np.pad(
                        v, ((0, 0), (0, n_samples - v.shape[-1])), mode="constant"
                    )
                else:
                    raise ValueError("audio chunk too short")

        return audio

    def _target_dbfs(self, audio):
        return 10.0 * np.log10(np.mean(np.square(np.abs(audio))) + 1e-6)

    def _chunk_and_check_dbfs_threshold(self, audio_, target_stem, threshold):
        target_audio = audio_[target_stem]

        for _ in range(self.max_dbfs_tries):
            start, end = self._get_start_end(target_audio)
            taudio = target_audio[..., start:end]
            dbfs = self._target_dbfs(taudio)
            if dbfs > threshold:
                return self._chunk_audio(audio_, start, end)

        return None

    def _chunk_and_check_dbfs(self, audio_, target_stem):
        out = self._chunk_and_check_dbfs_threshold(
            audio_, target_stem, self.min_target_dbfs
        )

        if out is not None:
            return out

        out = self._chunk_and_check_dbfs_threshold(
            audio_, target_stem, self.min_target_dbfs + self.min_target_dbfs_step
        )

        if out is not None:
            return out

        start, end = self._get_start_end(audio_[target_stem])
        audio = self._chunk_audio(audio_, start, end)

        return audio


class MoisesDBRandomChunkRandomQueryDataset(MoisesDBBaseChunkedQueryDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4.0,
        query_size_seconds: float = 1.0,
        min_query_dbfs: float = -40.0,
        min_target_dbfs: float = -36.0,
        min_target_dbfs_step: float = -12.0,
        max_dbfs_tries: int = 10,
        top_k_instrument: int = 10,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap: bool = True,
        allowed_stems=None,
        query_file: str = "query",
        augment=None,
    ) -> None:
        super().__init__(
            data_root,
            split,
            return_stems=False,
            npy_memmap=npy_memmap,
            recompute_mixture=augment is not None,
            chunk_size_seconds=chunk_size_seconds,
            query_size_seconds=query_size_seconds,
            query_file=query_file,
            mixture_stem=mixture_stem,
            min_query_dbfs=min_query_dbfs,
            min_target_dbfs=min_target_dbfs,
            min_target_dbfs_step=min_target_dbfs_step,
            max_dbfs_tries=max_dbfs_tries,
            top_k_instrument=top_k_instrument,
            allowed_stems=allowed_stems,
            use_own_query=use_own_query,
            augment=augment,
        )

        self.target_length = target_length

    def __len__(self) -> int:
        return self.target_length

    def _get_start_end(self, audio):
        return self._get_random_start_end(audio)

    def _get_audio(
        self,
        stems,
        identifier: Dict[str, Any],
        check_dbfs: bool = True,
        no_target: bool = False,
    ):
        target_stem = stems[0] if not no_target else None

        if self.augment is not None:
            stems_ = self._choose_stems_for_augment(identifier, target_stem)
        else:
            stems_ = stems

        audio = {}
        for stem in stems_:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        # audio_ = {k: v.copy() for k, v in audio.items()}
        audio_ = audio

        if check_dbfs:
            assert target_stem is not None
            audio = self._chunk_and_check_dbfs(audio_, target_stem)
        else:
            first_key = next(iter(audio_.keys()))
            start, end = self._get_start_end(audio_[first_key])
            audio = self._chunk_audio(audio_, start, end)

        if self.augment is not None:
            assert "mixture" not in audio
            audio = self._augment(audio, target_stem)
            assert "mixture" in audio

        return audio

    def __getitem__(self, index: int):
        mix_identifier = self.get_identifier(index)
        mix_stems = self.song_to_stem[mix_identifier["song_id"]]

        if self.use_own_query:
            query_id = mix_identifier["song_id"]
            query_identifier = dict(song_id=query_id)
            possible_stem = mix_stems

            assert len(possible_stem) > 0

            zero_target = False
        else:
            query_id = random.choice(self.queriable_songs)
            query_identifier = dict(song_id=query_id)
            query_stems = self.song_to_stem[query_id]
            possible_stem = list(set(mix_stems) & set(query_stems))

            if len(possible_stem) == 0:
                possible_stem = query_stems
                zero_target = True
                # print(f"Mix {mix_identifier['song_id']} and query {query_id} have no common stems.")
                # return self.__getitem__(index + 1)
            else:
                zero_target = False

        assert len(possible_stem) > 0, (
            f"{mix_identifier['song_id']} and {query_id} have no common stems. zero target is {zero_target}"
        )
        stem = random.choice(possible_stem)

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

        query = self.get_query_stem(stem=stem, identifier=query_identifier)
        query = query.copy()

        assert mixture.shape[-1] == self.chunk_size_samples
        assert query.shape[-1] == self.query_size_samples
        assert sources["target"].shape[-1] == self.chunk_size_samples

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier,
                "query": query_identifier,
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBRandomChunkBalancedRandomQueryDataset(
    MoisesDBRandomChunkRandomQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4,
        query_size_seconds: float = 1,
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
        query_file: str = "query",
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

        self.stem_to_n_songs = {k: len(v) for k, v in self.stem_to_song.items()}
        self.trainable_stems = [k for k, v in self.stem_to_n_songs.items() if v > 1]
        self.n_allowed_stems = len(self.allowed_stems)

    def __getitem__(self, index: int):
        stem = self.allowed_stems[index % self.n_allowed_stems]
        song_ids_with_stem = self.stem_to_song[stem]

        song_id = song_ids_with_stem[index % self.stem_to_n_songs[stem]]

        mix_identifier = dict(song_id=song_id)

        audio = self._get_audio(
            [stem, self.mixture_stem], identifier=mix_identifier, check_dbfs=True
        )
        mixture = audio[self.mixture_stem].copy()

        if self.use_own_query:
            query_id = song_id
            query_identifier = dict(song_id=query_id)
        else:
            query_id = random.choice(song_ids_with_stem)
            query_identifier = dict(song_id=query_id)

        query = self.get_query_stem(stem=stem, identifier=query_identifier)
        query = query.copy()

        sources = {"target": audio[stem].copy()}

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier,
                "query": query_identifier,
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBDeterministicChunkDeterministicQueryDataset(
    MoisesDBRandomChunkRandomQueryDataset
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
        query_file: str = "query",
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            target_length=None,
            chunk_size_seconds=chunk_size_seconds,
            query_size_seconds=query_size_seconds,
            min_query_dbfs=min_query_dbfs,
            top_k_instrument=top_k_instrument,
            mixture_stem=mixture_stem,
            use_own_query=use_own_query,
            npy_memmap=npy_memmap,
            allowed_stems=allowed_stems,
            query_file=query_file,
        )

        if hop_size_seconds is None:
            hop_size_seconds = chunk_size_seconds

        self.chunk_hop_size_seconds = hop_size_seconds

        self.chunk_hop_size_samples = int(hop_size_seconds * self.fs)

        self.n_queries_per_chunk = n_queries_per_chunk

        self._overwrite = False

        self.query_tuples = self.find_query_tuples_or_generate()
        self.n_chunks = len(self.query_tuples)

    def __len__(self) -> int:
        return self.n_chunks

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}

        for stem in stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        start = identifier["chunk_start"]
        end = start + self.chunk_size_samples
        audio = self._chunk_audio(audio, start, end)

        return audio

    def find_query_tuples_or_generate(self):
        query_path = os.path.join(self.data_path, "queries")
        val_folds = "-".join(map(str, self.folds))

        path_so_far = os.path.join(query_path, val_folds)

        if not os.path.exists(path_so_far):
            return self.generate_index()

        chunk_specs = f"chunk{self.chunk_size_samples}-hop{self.chunk_hop_size_samples}"
        path_so_far = os.path.join(path_so_far, chunk_specs)

        if not os.path.exists(path_so_far):
            return self.generate_index()

        query_specs = f"query{self.query_size_samples}-n{self.n_queries_per_chunk}"
        path_so_far = os.path.join(path_so_far, query_specs)

        if not os.path.exists(path_so_far):
            return self.generate_index()

        if self.top_k_instrument is not None:
            path_so_far = os.path.join(
                path_so_far, f"queries-top{self.top_k_instrument}.csv"
            )
        else:
            if len(self.allowed_stems) > 5:
                allowed_stems = (
                    str(len(self.allowed_stems))
                    + "stems:"
                    + ":".join([k[0] for k in self.allowed_stems if k != "mixture"])
                )
            else:
                allowed_stems = ":".join(self.allowed_stems)

            path_so_far = os.path.join(path_so_far, f"queries-{allowed_stems}.csv")

        if not os.path.exists(path_so_far):
            return self.generate_index()

        print(f"Loading query tuples from {path_so_far}")

        return pd.read_csv(path_so_far)

    def _get_index_path(self):
        query_root = os.path.join(self.data_path, "queries")
        val_folds = "-".join(map(str, self.folds))
        chunk_specs = f"chunk{self.chunk_size_samples}-hop{self.chunk_hop_size_samples}"
        query_specs = f"query{self.query_size_samples}-n{self.n_queries_per_chunk}"
        query_dir = os.path.join(query_root, val_folds, chunk_specs, query_specs)

        if self.top_k_instrument is not None:
            query_path = os.path.join(
                query_dir, f"queries-top{self.top_k_instrument}.csv"
            )
        else:
            if len(self.allowed_stems) > 5:
                allowed_stems = (
                    str(len(self.allowed_stems))
                    + "stems:"
                    + ":".join([k[0] for k in self.allowed_stems if k != "mixture"])
                )
            else:
                allowed_stems = ":".join(self.allowed_stems)
            query_path = os.path.join(query_dir, f"queries-{allowed_stems}.csv")

        if not self._overwrite:
            assert not os.path.exists(query_path), (
                f"Query path {query_path} already exists."
            )

        os.makedirs(query_dir, exist_ok=True)

        return query_path

    def generate_index(self):
        query_path = self._get_index_path()

        durations = pd.read_csv(os.path.join(self.data_path, "durations.csv"))
        durations = (
            durations[["song_id", "duration"]]
            .set_index("song_id")["duration"]
            .to_dict()
        )

        tuples = []

        stems_without_queries = defaultdict(list)

        for i, song_id in tqdm(enumerate(self.files), total=len(self.files)):
            song_duration = durations[song_id]
            mix_stems = self.song_to_stem[song_id]

            n_mix_chunks = math.floor(
                (song_duration - self.chunk_size_seconds) / self.chunk_hop_size_seconds
            )

            for stem in mix_stems:
                possible_queries = self.stem_to_song[stem]
                if song_id in possible_queries:
                    possible_queries.remove(song_id)

                if len(possible_queries) == 0:
                    stems_without_queries[song_id].append(stem)
                    continue

                for k in tqdm(range(n_mix_chunks), desc=f"song{i + 1}/{stem}"):
                    mix_chunk_start = int(k * self.chunk_hop_size_samples)

                    for j in range(self.n_queries_per_chunk):
                        query = random.choice(possible_queries)

                        tuples.append(
                            dict(
                                mix=song_id,
                                query=query,
                                stem=stem,
                                mix_chunk_start=mix_chunk_start,
                            )
                        )

        if len(stems_without_queries) > 0:
            print("Stems without queries:")
            for song_id, stems in stems_without_queries.items():
                print(f"{song_id}: {stems}")

        tuples = pd.DataFrame(tuples)

        print(
            f"Generating query tuples for {self.split} set with {len(tuples)} tuples."
        )
        print(f"Saving query tuples to {query_path}")

        tuples.to_csv(query_path, index=False)

        return tuples

    def index_to_identifiers(self, index: int) -> Tuple[str, str, str, int]:
        row = self.query_tuples.iloc[index]
        mix_id = row["mix"]

        if self.use_own_query:
            query_id = mix_id
        else:
            query_id = row["query"]

        stem = row["stem"]
        mix_chunk_start = row["mix_chunk_start"]

        return mix_id, query_id, stem, mix_chunk_start

    def __getitem__(self, index: int):
        mix_id, query_id, stem, mix_chunk_start = self.index_to_identifiers(index)

        mix_identifier = dict(song_id=mix_id, chunk_start=mix_chunk_start)
        query_identifier = dict(song_id=query_id)

        audio = self._get_audio([stem, self.mixture_stem], identifier=mix_identifier)
        query = self.get_query_stem(stem=stem, identifier=query_identifier)

        mixture = audio[self.mixture_stem].copy()
        sources = {"target": audio[stem].copy()}
        query = query.copy()

        assert mixture.shape[-1] == self.chunk_size_samples
        # print(query.shape[-1], self.query_size_samples)
        assert query.shape[-1] == self.query_size_samples
        assert sources["target"].shape[-1] == self.chunk_size_samples

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier,
                "query": query_identifier,
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBFullTrackTestQueryDataset(MoisesDBFullTrackDataset):
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
            data_root=data_root,
            split=split,
            npy_memmap=npy_memmap,
            recompute_mixture=False,
            query_file=query_file,
        )

        self.use_own_query = use_own_query

        self.allowed_stems = allowed_stems

        test_indices = pd.read_csv(os.path.join(data_root, "test_indices.csv"))

        test_indices = test_indices[test_indices.stem.isin(self.allowed_stems)]

        self.test_indices = test_indices

        self.length = len(self.test_indices)

    def __len__(self) -> int:
        return self.length

    def index_to_identifiers(self, index: int) -> Tuple[str, str, str]:
        row = self.test_indices.iloc[index]
        mix_id = row["song_id"]
        if self.use_own_query:
            query_id = mix_id
        else:
            query_id = row["query_id"]
        stem = row["stem"]

        return mix_id, query_id, stem

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}

        for stem in stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        return audio

    def __getitem__(self, index: int):
        mix_id, query_id, stem = self.index_to_identifiers(index)

        mix_identifier = dict(song_id=mix_id)

        query_identifier = dict(song_id=query_id)

        audio = self._get_audio([stem, "mixture"], identifier=mix_identifier)
        query = self.get_query_stem(stem=stem, identifier=query_identifier)

        mixture = audio["mixture"].copy()
        sources = {stem: audio[stem].copy()}
        query = query.copy()

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier["song_id"],
                "query": query_identifier["song_id"],
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBRandomChunkDualRandomQueryDataset(MoisesDBRandomChunkRandomQueryDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4,
        query_size_seconds: float = 1,
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
        query_file: str = "query",
        augment=None,
    ) -> None:
        assert use_own_query

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

        self.queriable_stems = [k for k, v in self.stem_to_song.items() if len(v) >= 2]

    def _chunk_audio(self, audio, start, end):
        # assert "target2" in audio, "check chunk"
        n_samples = end - start

        audio = {k: v[..., start:end] for k, v in audio.items()}

        for k, v in audio.items():
            if v.shape[-1] < n_samples:
                audio[k] = np.pad(
                    v, ((0, 0), (0, n_samples - v.shape[-1])), mode="constant"
                )

        return audio

    def _get_audio(
        self,
        stems,
        identifier: Dict[str, Any],
        query_identifier: Dict[str, Any],
        check_dbfs: bool = True,
        no_target: bool = False,
    ):
        target_stem = stems[0] if not no_target else None

        if self.augment is not None:
            stems_ = self._choose_stems_for_augment(identifier, target_stem)
        else:
            stems_ = stems

        audio = {}
        for stem in stems_:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        audio["target2"] = self.get_full_stem(
            stem=target_stem, identifier=query_identifier
        )

        audio_ = {k: v.copy() for k, v in audio.items()}
        assert "target2" in audio_, "check 1"

        if check_dbfs:
            assert "target2" in audio_, "check if"
            assert target_stem is not None
            audio = self._chunk_and_check_dbfs(audio_, target_stem)
        else:
            assert "target2" in audio_, "check else"
            first_key = next(iter(audio_.keys()))
            start, end = self._get_start_end(audio_[first_key])
            assert "target2" in audio_, "check else 2"
            audio = self._chunk_audio(audio_, start, end)

        if self.augment is not None:
            assert "mixture" not in audio
            assert "target2" in audio, "check augment"
            audio = self._augment(audio, target_stem)
            assert "mixture" in audio

        assert "target2" in audio, "check 2"

        return audio

    def _augment(self, audio, target_stem):
        stack_audio = np.stack(list(audio.values()), axis=0)
        aug_audio = self.augment(torch.from_numpy(stack_audio)).numpy()
        mixture = np.sum(aug_audio, axis=0)

        out = {
            "mixture": mixture,
        }

        target_idx = list(audio.keys()).index(target_stem)
        out[target_stem] = aug_audio[target_idx]

        target2_idx = list(audio.keys()).index("target2")
        out["target2"] = aug_audio[target2_idx]

        return out

    def __getitem__(self, index: int):
        mix_identifier = self.get_identifier(index)
        mix_stems = self.song_to_stem[mix_identifier["song_id"]]

        query_id = mix_identifier["song_id"]
        query_identifier = dict(song_id=query_id)
        possible_stem = list(set(mix_stems) & set(self.queriable_stems))

        assert len(possible_stem) > 0

        stem = random.choice(possible_stem)

        possible_query2_songs = self.stem_to_song[stem]
        possible_query2_songs = [s for s in possible_query2_songs if s != query_id]
        assert len(possible_query2_songs) > 0
        query2_id = random.choice(possible_query2_songs)
        assert query2_id != query_id
        query2_identifier = dict(song_id=query2_id)

        audio = self._get_audio(
            [stem, self.mixture_stem],
            identifier=mix_identifier,
            query_identifier=query2_identifier,
            check_dbfs=True,
        )

        mixture = audio[self.mixture_stem].copy()
        sources = {"target1": audio[stem].copy(), "target2": audio["target2"].copy()}

        query1 = self.get_query_stem(stem=stem, identifier=query_identifier)
        query1 = query1.copy()

        query2 = self.get_query_stem(stem=stem, identifier=query2_identifier)
        query2 = query2.copy()

        assert mixture.shape[-1] == self.chunk_size_samples
        assert query1.shape[-1] == self.query_size_samples
        assert query2.shape[-1] == self.query_size_samples
        assert sources["target1"].shape[-1] == self.chunk_size_samples, sources[
            "target1"
        ].shape[-1]
        assert sources["target2"].shape[-1] == self.chunk_size_samples, sources[
            "target2"
        ].shape[-1]

        return dual_input_dict(
            mixture=mixture,
            sources=sources,
            query1=query1,
            query2=query2,
            metadata={
                "mix": mix_identifier,
                "query1": query_identifier,
                "query2": query2_identifier,
                "stem": stem,
            },
            modality="audio",
        )

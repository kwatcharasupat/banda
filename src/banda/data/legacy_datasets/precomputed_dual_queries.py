import ast
import math
import os
import random
import pandas as pd
import torch
from bandx.data.moisesdb.legacy_dataset import (
    COARSE_LEVEL_INSTRUMENTS,
    FINE_LEVEL_INSTRUMENTS,
    FINE_TO_COARSE,
    COARSE_TO_FINE,
)
from bandx.data.moisesdb.datasets.precomputed_query import (
    MoisesDBRandomChunkPrecomputedQueryDataset,
)
from bandx.types import dual_input_dict


import librosa
import numpy as np


from collections import defaultdict
from itertools import combinations
from typing import Any, Dict


class MoisesDBRandomChunkPrecomputedRandomPairedQueryDataset(
    MoisesDBRandomChunkPrecomputedQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        within_coarse: bool = True,
        outside_coarse: bool = True,
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
            top_k_instrument=None,
            mixture_stem=mixture_stem,
            use_own_query=None,
            npy_memmap=npy_memmap,
            allowed_stems=FINE_LEVEL_INSTRUMENTS,
            query_file=None,
            augment=augment,
        )

        self.within_coarse = within_coarse
        self.outside_coarse = outside_coarse

        assert bool(within_coarse) + bool(outside_coarse) <= 1

        self.stem_to_song = {
            k: list(v) for k, v in self.stem_to_song.items() if len(v) >= 2
        }

        self.allowed_stems = set(self.allowed_stems) & set(self.stem_to_song.keys())
        self.allowed_stems = {
            s
            for s in self.allowed_stems
            if s in COARSE_LEVEL_INSTRUMENTS
            or len(set(COARSE_TO_FINE[FINE_TO_COARSE[s]]) & self.allowed_stems) > 1
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

        if target_stem in COARSE_LEVEL_INSTRUMENTS:
            song_coarse_stems = self.song_to_coarse_stems[song_id]

            song_stems = []

            for coarse_stem in song_coarse_stems:
                if coarse_stem == target_stem:
                    continue

                fine_in_coarse = COARSE_TO_FINE[coarse_stem] & set(song_fine_stems)
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

        if self.within_coarse:
            coarse1 = FINE_TO_COARSE[stem1]
            possible_stem2 = list(COARSE_TO_FINE[coarse1] & set(self.allowed_stems))
        elif self.outside_coarse:
            coarse1 = FINE_TO_COARSE[stem1]
            possible_stem2 = list(self.allowed_stems - COARSE_TO_FINE[coarse1])
        else:
            possible_stem2 = list(self.allowed_stems).copy()

        if stem1 in possible_stem2:
            possible_stem2 = [s for s in possible_stem2 if s != stem1]

        stem2 = random.choice(possible_stem2)

        if stem2 in song1_stems:
            song1_stems = [s for s in song1_stems if s != stem2]

        possible_song2 = self.stem_to_song[stem2]
        if song1_id in possible_song2:
            possible_song2 = [s for s in possible_song2 if s != song1_id]
        song2_id = random.choice(possible_song2)

        song1_audio = self._get_audio(
            stem1,
            song1_stems,
            {"song_id": song1_id, "index": index, "stem_order": 1},
            target_name="target1",
        )

        song2_audio = self._get_audio(
            stem2,
            [],
            {"song_id": song2_id, "index": index, "stem_order": 2},
            target_name="target2",
        )

        song1_audio.update(song2_audio)

        audio = song1_audio

        if self.augment is not None:
            audio = self._augment(audio, ["target1", "target2"])
            mixture = audio.pop("mixture")
            sources = audio
        else:
            mixture = sum(audio.values())
            sources = {"target1": audio["target1"], "target2": audio["target2"]}

        query1 = self.get_query_embedding(stem=stem1, song_id=song1_id)
        query2 = self.get_query_embedding(stem=stem2, song_id=song2_id)

        # print('got')
        return dual_input_dict(
            mixture=mixture,
            sources=sources,
            query1={"passt": query1},
            query2={"passt": query2},
            metadata={
                "song1_id": song1_id,
                "stem1": stem1,
                "song2_id": song2_id,
                "stem2": stem2,
            },
            modality="audio",
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

    def _get_audio(
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


class MoisesDBDeterministicChunkPrecomputedDeterministicPairedQueryDataset(
    MoisesDBRandomChunkPrecomputedRandomPairedQueryDataset
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
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            target_length=target_length,
            within_coarse=False,
            outside_coarse=False,
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
        )

        index = []

        stem_combinations = list(combinations(self.allowed_stems, 2))

        file_combi_per_stem_combi = max(1, self.target_length // len(stem_combinations))

        used_files1 = set()
        used_files2 = set()

        counter_by_stem1 = defaultdict(lambda: 0)
        counter_by_stem2 = defaultdict(lambda: 0)

        for stem1, stem2 in stem_combinations:
            file1s = self.stem_to_song[stem1]
            file2s = self.stem_to_song[stem2].copy()

            n_combis = min(len(file1s) * len(file2s), file_combi_per_stem_combi)

            for _ in range(n_combis):
                file1s = self.stem_to_song[stem1]
                file2s = self.stem_to_song[stem2].copy()

                unused_file1s = sorted(set(file1s) - used_files1)
                if len(unused_file1s) > 0:
                    file1 = unused_file1s[0]
                else:
                    file1 = file1s[counter_by_stem1[stem1] % len(file1s)]
                    counter_by_stem1[stem1] += 1

                used_files1.add(file1)
                if len(used_files1) == len(self.files):
                    used_files1.clear()

                if file1 in file2s:
                    file2s.remove(file1)

                    if len(file2s) == 0:
                        continue

                unused_file2s = sorted(set(file2s) - used_files2)
                if len(unused_file2s) > 0:
                    file2 = unused_file2s[0]
                else:
                    file2 = file2s[counter_by_stem2[stem2] % len(file2s)]
                    counter_by_stem2[stem2] += 1

                index.append(
                    {
                        "song1_id": file1,
                        "song2_id": file2,
                        "stem1": stem1,
                        "stem2": stem2,
                    }
                )

        self.index = index
        self.pair_start_time = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        # print(index, end=" ")

        identifiers = self.index[index]

        song1_id = identifiers["song1_id"]
        song2_id = identifiers["song2_id"]
        stem1 = identifiers["stem1"]
        stem2 = identifiers["stem2"]

        song1_stems = self.get_nontarget_stems(song1_id, stem1)

        if stem2 in song1_stems:
            song1_stems = [s for s in song1_stems if s != stem2]

        song1_audio = self._get_audio(
            stem1, song1_stems, {"song_id": song1_id, "index": index, "stem_order": 1}
        )
        song2_audio = self._get_audio(
            stem2, [], {"song_id": song2_id, "index": index, "stem_order": 2}
        )

        target1 = song1_audio.pop("target")
        target2 = song2_audio["target"]

        audio = {**song1_audio, "target1": target1, "target2": target2}

        if self.augment is not None:
            audio = self._augment(audio, ["target1", "target2"])
            mixture = audio.pop("mixture")
            sources = audio
        else:
            mixture = sum(audio.values())
            sources = {"target1": target1, "target2": target2}

        query1 = self.get_query_embedding(stem=stem1, song_id=song1_id)
        query2 = self.get_query_embedding(stem=stem2, song_id=song2_id)

        # print('got')
        return dual_input_dict(
            mixture=mixture,
            sources=sources,
            query1={"passt": query1},
            query2={"passt": query2},
            metadata={
                "song1_id": song1_id,
                "stem1": stem1,
                "song2_id": song2_id,
                "stem2": stem2,
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


class MoisesDBRandomChunkPrecomputedPairedQueryDataset(
    MoisesDBRandomChunkPrecomputedQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        within_coarse: bool = True,
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
            top_k_instrument=None,
            mixture_stem=mixture_stem,
            use_own_query=None,
            npy_memmap=npy_memmap,
            allowed_stems=None,
            query_file=None,
            augment=augment,
        )

        df = pd.read_csv(os.path.join(data_root, "pairs", f"{split}.csv"))

        # print(df.columns)
        if within_coarse:
            df = df[df.within_coarse].reset_index()

        df.allowed_song1_stems = df.allowed_song1_stems.apply(ast.literal_eval)

        self.pair_indices = df

        # print(self.pair_indices)

        self.true_length = len(self.pair_indices)

        if split == "train":
            assert augment

    def __len__(self) -> int:
        return self.target_length

    def get_query_embedding(self, stem, song_id):
        path = os.path.join(self.data_path, "passt-full")
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            query = np.load(
                os.path.join(path, song_id, f"{stem}.passt.stats.npy"), mmap_mode="r"
            )
            assert query.shape == (2, 768)
        else:
            raise NotImplementedError

        return query

    def __getitem__(self, index: int):
        # print(index, end=" ")
        pair = self.pair_indices.iloc[index % self.true_length]

        song1_audio = self._get_audio(
            pair.stem1,
            pair.allowed_song1_stems,
            {"song_id": pair.song_id1, "index": index, "stem_order": 1},
        )
        song2_query_audio = self._get_audio(
            pair.stem2, [], {"song_id": pair.song_id2, "index": index, "stem_order": 2}
        )

        target1 = song1_audio.pop("target")
        target2 = song2_query_audio["target"]

        audio = {**song1_audio, "target1": target1, "target2": target2}

        # print(audio.keys())

        if self.augment is not None:
            audio = self._augment(audio, ["target1", "target2"])
            mixture = audio.pop("mixture")
            sources = audio
        else:
            mixture = sum(audio.values())
            sources = {"target1": target1, "target2": target2}

        query1 = self.get_query_embedding(
            stem=pair.stem1,
            song_id=pair.song_id1,
        )
        query2 = self.get_query_embedding(
            stem=pair.stem2,
            song_id=pair.song_id2,
        )

        metadata = pair.to_dict()
        metadata["allowed_song1_stems"] = ", ".join(metadata["allowed_song1_stems"])

        # print('got')
        return dual_input_dict(
            mixture=mixture,
            sources=sources,
            query1={"passt": query1},
            query2={"passt": query2},
            metadata=metadata,
            modality="audio",
        )

    def _augment(self, audio, target_stems):
        stack_audio = np.stack(list(audio.values()), axis=0)
        aug_audio = self.augment(torch.from_numpy(stack_audio)).numpy()
        mixture = np.sum(aug_audio, axis=0)

        out = {
            "mixture": mixture,
        }

        for target_stem in target_stems:
            target_idx = list(audio.keys()).index(target_stem)
            out[target_stem] = aug_audio[target_idx]

        return out

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

        if check_dbfs:
            assert target_stem is not None
            audio = self._chunk_and_check_dbfs(audio_, "target")
        else:
            first_key = next(iter(audio_.keys()))
            start, end = self._get_start_end(audio_[first_key])
            audio = self._chunk_audio(audio_, start, end)

        return audio


class MoisesDBDeterministicChunkPrecomputedPairedQueryDataset(
    MoisesDBRandomChunkPrecomputedPairedQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        within_coarse: bool = True,
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
        **kwargs,
    ) -> None:
        super().__init__(
            data_root,
            split,
            target_length,
            within_coarse,
            chunk_size_seconds,
            query_size_seconds,
            round_query,
            min_query_dbfs,
            min_target_dbfs,
            min_target_dbfs_step,
            max_dbfs_tries,
            mixture_stem,
            npy_memmap,
            augment,
        )

        if self.target_length < self.true_length:
            dfg = self.pair_indices.groupby(["stem1", "stem2"])

            # TODO: fix this to skip over song1 too!

            group_idx = dfg.groups
            n_per_group = {k: len(v) for k, v in group_idx.items()}
            n_groups = len(n_per_group)

            n_needed_per_group = self.target_length / n_groups

            n_skip_per_group = {
                k: max(1, math.floor(v / n_needed_per_group))
                for k, v in n_per_group.items()
            }

            indices = []

            for group, group_indices in group_idx.items():
                # print(group_indices)
                using = list(group_indices[:: n_skip_per_group[group]])
                indices += using

            indices = sorted(indices)

            self.pair_indices = self.pair_indices.loc[indices]

            self.true_length = len(self.pair_indices)

        self.pair_start_time = {}

    def __len__(self) -> int:
        return self.true_length

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

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from bandx.data.moisesdb.legacy_dataset import (
    ALL_LEVEL_INSTRUMENTS,
    COARSE_LEVEL_INSTRUMENTS,
    FINE_LEVEL_INSTRUMENTS,
    MoisesDBFullTrackDataset,
)


def remove_empty_and_sort(d):
    return {k: sorted(v) for k, v in d.items() if len(v) > 0}


class MoisesDBRawStemsDataset(MoisesDBFullTrackDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        allowed_stems=None,
        npy_memmap: bool = True,
        fine_only: bool = True,
        coarse_only: bool = False,
        rank=None,
        n_jobs=None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            npy_memmap=npy_memmap,
            recompute_mixture=False,
        )

        assert fine_only ^ coarse_only, (
            "Only one of fine_only or coarse_only can be True"
        )

        self.fine_only = fine_only
        self.coarse_only = coarse_only

        if allowed_stems is None:
            if fine_only:
                self.allowed_stems = FINE_LEVEL_INSTRUMENTS
            elif coarse_only:
                self.allowed_stems = COARSE_LEVEL_INSTRUMENTS
            else:
                self.allowed_stems = ALL_LEVEL_INSTRUMENTS
        else:
            self.allowed_stems = allowed_stems

        self.initialize_lookups()

    def initialize_lookups(self) -> None:
        self.song_to_all_stems = {
            k: list(set(v) & set(ALL_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }
        self.song_to_all_stems = remove_empty_and_sort(self.song_to_all_stems)

        self.song_to_fine_stems = {
            k: list(set(v) & set(FINE_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }
        self.song_to_fine_stems = remove_empty_and_sort(self.song_to_fine_stems)

        self.song_to_coarse_stems = {
            k: list(set(v) & set(COARSE_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }
        self.song_to_coarse_stems = remove_empty_and_sort(self.song_to_coarse_stems)

        self.song_to_allowed_all_stems = {
            k: list(set(v) & set(self.allowed_stems))
            for k, v in self.song_to_all_stems.items()
        }
        self.song_to_allowed_all_stems = remove_empty_and_sort(
            self.song_to_allowed_all_stems
        )

        self.song_to_allowed_coarse_stems = {
            k: list(set(v) & set(self.allowed_stems))
            for k, v in self.song_to_coarse_stems.items()
        }
        self.song_to_allowed_coarse_stems = remove_empty_and_sort(
            self.song_to_allowed_coarse_stems
        )

        self.song_to_allowed_fine_stems = {
            k: list(set(v) & set(self.allowed_stems))
            for k, v in self.song_to_fine_stems.items()
        }
        self.song_to_allowed_fine_stems = remove_empty_and_sort(
            self.song_to_allowed_fine_stems
        )

        if self.fine_only:
            self.files = list(self.song_to_allowed_fine_stems.keys())
        elif self.coarse_only:
            self.files = list(self.song_to_allowed_coarse_stems.keys())
        else:
            self.files = list(self.song_to_allowed_all_stems.keys())

        self.true_length = len(self.files)

    def __len__(self) -> int:
        return self.true_length

    def get_stem(self, *, stem: str, identifier) -> torch.Tensor:
        song_id = identifier["song_id"]
        path = os.path.join(self.data_path, "npy2", song_id)

        assert self.npy_memmap

        if os.path.exists(os.path.join(path, f"{stem}.npy")):
            audio = np.load(os.path.join(path, f"{stem}.npy"), mmap_mode="r")
        else:
            raise FileNotFoundError(f"{song_id, stem}")

        return audio

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}
        for stem in stems:
            audio[stem] = self.get_stem(stem=stem, identifier=identifier)

        return audio

    def get_allowed_stems(self, identifier: Dict[str, Any]):
        if self.fine_only:
            return self.song_to_allowed_fine_stems[identifier["song_id"]]
        elif self.coarse_only:
            return self.song_to_allowed_coarse_stems[identifier["song_id"]]
        else:
            return self.song_to_allowed_all_stems[identifier["song_id"]]


class MoisesDBBaseChunkedRawStemsDataset(MoisesDBRawStemsDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 10.0,
        dbfs_threshold: float = -48.0,
        allowed_stems=None,
        npy_memmap: bool = True,
        fine_only: bool = True,
        coarse_only: bool = False,
        allow_pad: bool = False,
        # **kwargs,
    ) -> None:
        # print(kwargs)

        super().__init__(
            data_root=data_root,
            split=split,
            npy_memmap=npy_memmap,
            fine_only=fine_only,
            coarse_only=coarse_only,
            allowed_stems=allowed_stems,
        )

        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = int(chunk_size_seconds * self.fs)

        self.dbfs_threshold = dbfs_threshold

        self.allow_pad = allow_pad

    def _chunk_audio(self, audio, start, end):
        n_samples = end - start

        out = {}

        for stem, v in audio.items():
            chunked = v[..., start:end]
            if chunked.shape[-1] < n_samples:
                if self.allow_pad:
                    chunked = np.pad(
                        chunked,
                        ((0, 0), (0, n_samples - chunked.shape[-1])),
                        mode="constant",
                    )
                else:
                    continue

            if dbrms(chunked) < self.dbfs_threshold:
                continue

            out[stem] = chunked

        if len(out) == 0:
            return None

        return out

    def _get_start_end(self, audio, index=None):
        raise NotImplementedError

    def _get_random_start_end(self, audio):
        first_audio = next(iter(audio.values()))
        n_samples = first_audio.shape[-1]
        start = np.random.randint(0, n_samples - self.chunk_size_samples)
        end = start + self.chunk_size_samples

        return start, end

    def handle_empty_audio(self, index, identifier):
        raise NotImplementedError

    def __getitem__(self, index: int):
        identifier = self.get_identifier(index)
        stems = self.get_allowed_stems(identifier)
        audio = self._get_audio(stems, identifier=identifier)
        start, end = self._get_start_end(audio, index=index)
        audio = self._chunk_audio(audio, start, end)

        if audio is None:
            return self.handle_empty_audio(index, identifier)

        return {
            "sources": {
                k: {
                    "audio": v.copy(),
                }
                for k, v in audio.items()
            },
            "metadata": {
                "song_id": identifier["song_id"],
                "start": start,
                "end": end,
            },
        }


class MoisesDBRandomChunkedRawStemsDataset(MoisesDBBaseChunkedRawStemsDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 10.0,
        dbfs_threshold: float = -48.0,
        allowed_stems=None,
        npy_memmap: bool = True,
        fine_only: bool = True,
        coarse_only: bool = False,
        allow_pad: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            chunk_size_seconds=chunk_size_seconds,
            dbfs_threshold=dbfs_threshold,
            allowed_stems=allowed_stems,
            npy_memmap=npy_memmap,
            fine_only=fine_only,
            coarse_only=coarse_only,
            allow_pad=allow_pad,
        )

        self.target_length = target_length

    def __len__(self) -> int:
        return self.target_length

    def _get_start_end(self, audio, index=None):
        return self._get_random_start_end(audio)

    def handle_empty_audio(self, index, identifier):
        return self.__getitem__(index + 1)


class MoisesDBDeterministicChunkedRawStemsDataset(MoisesDBBaseChunkedRawStemsDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 10.0,
        hop_size_seconds: float = 10.0,
        dbfs_threshold: float = -48.0,
        allowed_stems=None,
        npy_memmap: bool = True,
        fine_only: bool = True,
        coarse_only: bool = False,
        target_length=None,
        allow_pad: bool = False,
        rank: int = 0,
        n_jobs: int = 1,
        return_audio: bool = True,
        return_passt: bool = not True,
        use_pca_passt: bool = not True,
        sort_by_n_stems: int = 1,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            chunk_size_seconds=chunk_size_seconds,
            dbfs_threshold=dbfs_threshold,
            npy_memmap=npy_memmap,
            fine_only=fine_only,
            coarse_only=coarse_only,
            allowed_stems=allowed_stems,
            allow_pad=allow_pad,
        )

        # raise NotImplementedError

        self.hop_size_seconds = hop_size_seconds
        self.hop_size_samples = int(hop_size_seconds * self.fs)

        self.target_length = target_length

        self.is_index_built = False

        self.initialize_index_lookup()

        # print(len(self.index_lookup))

        self.is_index_built = True

        self.rank = rank
        self.n_jobs = n_jobs

        if self.n_jobs > 1:
            self.index_lookup = self.index_lookup[self.rank :: self.n_jobs]

        self.return_audio = return_audio
        self.return_passt = return_passt
        self.use_pca_passt = use_pca_passt

        if self.return_passt and not self.return_audio:
            if self.use_pca_passt:
                self.passt_cache = os.path.expandvars("$DATA_ROOT/moisesdb/pca-128")
            else:
                self.passt_cache = os.path.expandvars("$DATA_ROOT/moisesdb/passt_cache")
            self.index_lookup = [
                (song_id, chunk_index)
                for song_id, chunk_index in self.index_lookup
                if os.path.exists(
                    os.path.expandvars(
                        os.path.join(
                            self.passt_cache,
                            self.split,
                            song_id,
                            f"{chunk_index:04d}.npz",
                        )
                    )
                )
            ]

            if abs(sort_by_n_stems) > 0:
                n_stems = [
                    np.load(
                        os.path.expandvars(
                            os.path.join(
                                self.passt_cache,
                                self.split,
                                song_id,
                                f"{chunk_index:04d}.npz",
                            )
                        )
                    )["passt"].shape[0]
                    for song_id, chunk_index in self.index_lookup
                ]

                order = np.argsort(n_stems)
                if sort_by_n_stems < 0:
                    order = order[::-1]

                self.index_lookup = [self.index_lookup[i] for i in order]

    def check_audio(self, audio, j):
        start = j * self.hop_size_samples
        end = start + self.chunk_size_samples

        audio = self._chunk_audio(audio, start, end)

        return audio

    def initialize_index_lookup(self) -> None:
        target_length_str = (
            "none" if self.target_length is None else f"{self.target_length}"
        )

        filename = os.path.join(
            self.data_path,
            f"index_lookup_{self.split}-{target_length_str}-{self.chunk_size_samples}-{self.hop_size_samples}.npy",
        )
        # print(filename)

        if os.path.exists(filename):
            df = pd.read_csv(filename)
            self.index_lookup = list(zip(df["song_id"], df["chunk_index"]))
            self.index_lookup = sorted(self.index_lookup)
            return

        if self.target_length is None:
            n_chunks_per_song = -1
            n_chunks = np.full(len(self.files), -1)
        else:
            n_chunks_per_song = self.target_length // len(self.files)
            n_missing = self.target_length - n_chunks_per_song * len(self.files)

            n_chunks = np.ones(len(self.files), dtype=int) * n_chunks_per_song
            n_chunks[:n_missing] += 1

        index_lookup = []

        n_samples_by_song = {}

        for i, song_id in enumerate(tqdm(self.files)):
            identifier = {"song_id": song_id}
            stems = self.get_allowed_stems(identifier)
            audio = self._get_audio(stems, identifier=identifier)

            n_samples = audio[next(iter(audio.keys()))].shape[-1]

            n_samples_by_song[song_id] = n_samples

        n_samples_by_song = dict(
            sorted(n_samples_by_song.items(), key=lambda item: item[1])
        )

        for i, (song_id, n_samples) in enumerate(tqdm(n_samples_by_song.items())):
            requested_chunks = n_chunks[i]

            identifier = {"song_id": song_id}
            stems = self.get_allowed_stems(identifier)
            audio = self._get_audio(stems, identifier=identifier)

            hop_size_samples = self.hop_size_samples

            n_chunks_total = (
                n_samples - self.chunk_size_samples
            ) // hop_size_samples + 1

            if self.target_length is not None:
                every_other = max(n_chunks_total // n_chunks[i], 1)
                chunk_indices = list(range(0, n_chunks_total, every_other))[
                    : n_chunks[i]
                ]
            else:
                chunk_indices = list(range(n_chunks_total))

            if requested_chunks > n_chunks_total:
                print(
                    f"Song {song_id} has {n_chunks_total} chunks before filtering, was requested {requested_chunks}"
                )

            provided_chunks = 0
            for j in chunk_indices:
                chunked_audio = None
                while chunked_audio is None:
                    chunked_audio = self.check_audio(audio, j)
                    if chunked_audio is None:
                        print(f"Empty audio at {song_id} chunk {j}", end=" ")
                        j += 1
                        if j in chunk_indices:
                            continue
                        if j >= n_chunks_total:
                            break
                        print(f"Incrementing to {j}")

                if chunked_audio is not None:
                    index_lookup.append((song_id, j))
                    provided_chunks += 1

            if provided_chunks < requested_chunks:
                if provided_chunks < n_chunks_total:
                    print(
                        f"Song {song_id} has {provided_chunks} chunks out of {min(requested_chunks, n_chunks_total)}"
                    )

                missing_chunks = requested_chunks - provided_chunks
                for k in range(i + 1, len(self.files)):
                    if n_chunks[k] == np.min(n_chunks[i + 1 :]):
                        n_chunks[k] += 1
                        missing_chunks -= 1
                    if missing_chunks == 0:
                        break

        df = pd.DataFrame(index_lookup, columns=["song_id", "chunk_index"])
        df = df.sort_values(by=["song_id", "chunk_index"])
        df.to_csv(filename, index=False)

        self.index_lookup = index_lookup

    # def _initialize_index_lookup(self):

    #     index_lookup = []

    #     # get the number of chunks for each song
    #     for song_id in tqdm(self.files):
    #         first_stem = self.get_allowed_stems({"song_id": song_id})[0]
    #         audio = self.get_stem(stem=first_stem, identifier={"song_id": song_id})
    #         n_samples = audio.shape[-1]
    #         n_chunks = (
    #             n_samples - self.chunk_size_samples
    #         ) // self.hop_size_samples + 1
    #         indices = [(song_id, i) for i in range(n_chunks)]
    #         index_lookup.append(indices)

    #     # cut the chunks to the target length
    #     if self.target_length is None:
    #         index_lookup = list(chain(*index_lookup))
    #     else:
    #         n_chunks_per_song = self.target_length // len(self.files)
    #         n_missing = self.target_length - n_chunks_per_song * len(self.files)

    #         unused_indices = []

    #         final_index_lookup = []

    #         for i, indices in enumerate(tqdm(index_lookup)):
    #             # print(indices)
    #             n_chunks = len(indices)
    #             if n_chunks > n_chunks_per_song:
    #                 every_other = n_chunks // n_chunks_per_song

    #                 using_indices = indices[::every_other][:n_chunks_per_song]
    #                 unused_indices.append(
    #                     [x for x in indices if x not in using_indices]
    #                 )
    #                 final_index_lookup.extend(
    #                     indices[::every_other][:n_chunks_per_song]
    #                 )
    #             elif n_chunks <= n_chunks_per_song:
    #                 final_index_lookup.extend(indices)
    #                 n_missing += n_chunks_per_song - n_chunks
    #                 unused_indices.append([])

    #         if n_missing > 0:
    #             while n_missing > 0:
    #                 all_empty = all([len(x) == 0 for x in unused_indices])
    #                 if all_empty:
    #                     break
    #                 for i, indices in enumerate(unused_indices):
    #                     if len(indices) > 0:
    #                         final_index_lookup.append(indices.pop())
    #                         n_missing -= 1
    #                         # print(n_missing)
    #                         if n_missing == 0:
    #                             break

    #     self.index_lookup = final_index_lookup

    def get_identifier(self, index):
        song_id, chunk_index = self.index_lookup[index]

        return {"song_id": song_id, "chunk_index": chunk_index}

    def __len__(self) -> int:
        return len(self.index_lookup)

    def _get_start_end(self, audio, index):
        assert index is not None

        start = index * self.hop_size_samples
        end = start + self.chunk_size_samples

        return start, end

    def handle_empty_audio(self, index, identifier):
        # audio = self._get_audio(self.get_allowed_stems(identifier), identifier=identifier)
        # start, end = self._get_start_end(audio, index=identifier["chunk_index"])
        # for stem, v in audio.items():
        #     print(stem, f"{dbrms(v[:, start:end]):.2f}", v.shape[0], f"{v.shape[1]/self.fs:.2f}")

        # mixture = self.get_stem(stem="mixture", identifier=identifier)

        # import torchaudio as ta
        # ta.save(
        #     f"empty_{identifier['song_id']}_{identifier['chunk_index']}.wav",
        #     torch.from_numpy(mixture).float(),
        #     self.fs,
        # )

        raise ValueError(
            f"Empty audio at {identifier['song_id']} {identifier['chunk_index']}"
        )

    def __getitem__(self, index: int):
        identifier = self.get_identifier(index)

        out_dict = {
            "metadata": {
                "song_id": identifier["song_id"],
                "chunk_index": identifier["chunk_index"],
            },
        }
        stems = self.get_allowed_stems(identifier)
        if self.return_audio:
            audio = self._get_audio(stems, identifier=identifier)
            start, end = self._get_start_end(audio, index=identifier["chunk_index"])

            audio = self._chunk_audio(audio, start, end)

            if audio is None:
                # print(start/self.fs, end/self.fs)
                return self.handle_empty_audio(index, identifier)

            out_dict["sources"] = {
                k: {
                    "audio": v.copy(),
                }
                for k, v in audio.items()
            }

        if self.return_passt:
            song_id = identifier["song_id"]
            chunk_index = identifier["chunk_index"]

            start = chunk_index * self.hop_size_samples
            end = start + self.chunk_size_samples

            if self.use_pca_passt:
                passt_cache = os.path.expandvars("$DATA_ROOT/moisesdb/pca-128")
            else:
                passt_cache = os.path.expandvars("$DATA_ROOT/moisesdb/passt_cache")

            cache_path = os.path.expandvars(
                os.path.join(passt_cache, self.split, song_id, f"{chunk_index:04d}.npz")
            )

            if os.path.exists(cache_path):
                passt = np.load(cache_path)
            else:
                # return out_dict
                raise FileNotFoundError(cache_path)

            out_dict["passt"] = passt["passt"]
            stems = passt["stems"]

            if not self.return_audio:
                out_dict["sources"] = {k: torch.empty((0,)) for k in stems}

        out_dict["metadata"]["start"] = start
        out_dict["metadata"]["end"] = end
        # print(out_dict)

        return out_dict


if __name__ == "__main__":
    ds = MoisesDBDeterministicChunkedRawStemsDataset(
        data_root=os.path.expandvars("$DATA_ROOT/moisesdb"),
        split="val",
        chunk_size_seconds=10,
        hop_size_seconds=5,
        dbfs_threshold=-48,
        npy_memmap=True,
        fine_only=True,
        coarse_only=False,
        target_length=1024,
        allow_pad=False,
    )

    for item in tqdm(ds):
        pass

from collections import defaultdict
from pprint import pprint
import random
import numpy as np
from banda.data.datasets.base import BaseRegisteredDataset, DatasetParams
from banda.data.datasources.base import BaseRegisteredDatasource, TrackIdentifier

from banda.data.datasources.moisesdb import MoisesDBStemWiseDatasource, StemIdentifier
from banda.data.item import SourceSeparationItem

import structlog

logger = structlog.get_logger(__name__)


class ChunkDatasetParams(DatasetParams):
    chunk_size_seconds: float


class RandomChunkDatasetParams(ChunkDatasetParams):
    max_dataset_size: int

    min_dbrms: float = -36
    dbrms_thresh_step: float = -1.2
    max_trial: int = 20

    cross_track_sampling: bool = False


class DeterministicChunkDatasetParams(ChunkDatasetParams):
    hop_size_seconds: float
    max_dataset_size: int | None = None


def _dbrms(x: np.ndarray, eps: float = 1e-12) -> float:
    if np.std(x) == 0:
        return -np.inf

    return 10 * np.log10(np.mean(np.square(x)) + eps)


class _ChunkDataset(BaseRegisteredDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        super().__init__(datasources=datasources, config=config)

    def _chunk_item(
        self, audio: np.ndarray | None, start_sample: int, *, pad: bool = False
    ) -> np.ndarray:
        end_sample = start_sample + self.chunk_size_samples
        out = audio[:, start_sample:end_sample]

        _, n_samples = out.shape

        if pad and n_samples < self.chunk_size_samples:
            padding = np.zeros((out.shape[0], self.chunk_size_samples - n_samples))
            out = np.concatenate([out, padding], axis=1)

        assert out.shape[1] == self.chunk_size_samples, (
            f"Chunked audio has incorrect shape: {out.shape}"
        )

        return out


class RandomChunkDataset(_ChunkDataset):
    config: RandomChunkDatasetParams

    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = RandomChunkDatasetParams.model_validate(config)
        super().__init__(datasources=datasources, config=config)

    def _cache_sizes(self):
        self.chunk_size_samples = int(self.config.chunk_size_seconds * self.config.fs)

    def __len__(self):
        return self.config.max_dataset_size

    def __getitem__(self, index: int):
        if self.config.cross_track_sampling:
            item_dict = self._load_cross_track(index)
        else:
            track_identifier = self._get_track_identifier(index)
            item_dict = self._load_audio(track_identifier)

        chunked_item_dict = self._chunk_and_augment(item_dict)

        return chunked_item_dict.model_dump()

    def _load_cross_track(self, index: int) -> SourceSeparationItem[np.ndarray]:
        track_identifier = self._get_track_identifier(index)
        sources = list(track_identifier.sources.keys())

        item_dict = self._load_audio(track_identifier, sources=[sources[0]])
        n_sources = len(sources)

        # logger.info(
        #     f"Cross-track sampling: Loading {n_sources} sources from different tracks."
        # )
        # logger.info(f"First source: {sources[0]} from {track_identifier.full_path}")

        for i in range(1, n_sources):
            next_index = random.randint(0, self.total_size - 1)
            next_track_identifier = self._get_track_identifier(next_index)
            next_source = sources[i]
            next_item_dict = self._load_audio(
                next_track_identifier, sources=[next_source]
            )
            item_dict.sources[next_source] = next_item_dict.sources[next_source]
            # logger.info(
            #     f"Next source: {next_source} from {next_track_identifier.full_path}"
            # )

        return item_dict

    def _chunk_and_augment(
        self, item_dict: SourceSeparationItem[np.ndarray], pre_pad: bool = False
    ) -> SourceSeparationItem[np.ndarray]:
        # Implement chunking logic here

        for source in item_dict.sources:
            audio = item_dict.sources[source]["audio"]

            if audio is None or len(audio) == 0:
                item_dict.sources[source]["audio"] = np.zeros(
                    shape=(self.config.n_channels, self.chunk_size_samples),
                    dtype=np.float32,
                )
                continue

            chunked_audio = [
                self._chunk_item_random(audio_component, pad=pre_pad)
                for audio_component in audio
            ]

            if self.premix_augmentation is not None:
                chunked_audio = [
                    self.premix_augmentation(
                        chunked_audio_component, sample_rate=self.config.fs
                    )
                    for chunked_audio_component in chunked_audio
                ]

            item_dict.sources[source]["audio"] = sum(chunked_audio)

        mixture = sum(
            item_dict.sources[source]["audio"] for source in item_dict.sources
        )
        item_dict.mixture = {"audio": mixture}

        return item_dict

    def _chunk_item_random(
        self,
        audio: np.ndarray,
        *,
        chunk_size_samples: int | None = None,
        trial_counter: int = 0,
        pad: bool = False,
    ) -> np.ndarray:
        _, n_samples = audio.shape

        if chunk_size_samples is None:
            chunk_size_samples = self.chunk_size_samples

        if n_samples < chunk_size_samples:
            if pad:
                audio = np.pad(
                    audio,
                    ((0, 0), (0, chunk_size_samples - n_samples)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                raise ValueError(
                    f"Audio is shorter than chunk size: {n_samples} < {chunk_size_samples}"
                )

        start_time = np.random.randint(0, max(1, n_samples - chunk_size_samples))
        out = self._chunk_item(audio, start_time)

        if trial_counter > self.config.max_trial:
            return out

        dbrms = _dbrms(out)
        thresh = self.config.min_dbrms + self.config.dbrms_thresh_step * trial_counter
        if dbrms > thresh:
            return out

        return self._chunk_item_random(audio, trial_counter=trial_counter + 1)


class DeterministicChunkDataset(_ChunkDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = DeterministicChunkDatasetParams.model_validate(config)
        if config.max_dataset_size is not None:
            logger.warning(
                f"Max dataset size is set. This may automatically increase hop size if the total number of chunks exceeds {config.max_dataset_size}."
            )
        super().__init__(datasources=datasources, config=config)

    def _cache_sizes(self):
        self.chunk_size_samples = int(self.config.chunk_size_seconds * self.config.fs)
        self.hop_size_samples = int(self.config.hop_size_seconds * self.config.fs)

    def _build_index_cache(self):
        track_durations = []
        track_identifiers = []
        for ds_idx, ds in enumerate(self.datasources):
            for track_idx, track in enumerate(ds):
                track: TrackIdentifier
                n_samples = track.duration_samples
                track_durations.append(n_samples)
                track_identifiers.append({"ds_idx": ds_idx, "track_idx": track_idx})

        self.track_durations = np.array(track_durations)
        self.track_identifiers = track_identifiers

        initial_chunk_counts = (
            self.track_durations - self.chunk_size_samples
        ) // self.hop_size_samples + 1
        initial_total_chunks = initial_chunk_counts.sum()

        self.effective_hop_size_samples = self.hop_size_samples

        if (
            self.config.max_dataset_size is not None
            and initial_total_chunks > self.config.max_dataset_size
        ):
            # Recalculate the effective hop size to achieve the target size
            target_n_chunks = self.config.max_dataset_size
            total_possible_hops = (self.track_durations - self.chunk_size_samples).sum()

            # The new hop size is calculated to distribute the hops across all tracks
            self.effective_hop_size_samples = max(
                1,
                int(
                    np.floor(
                        total_possible_hops
                        / (target_n_chunks - len(self.track_durations))
                    )
                ),
            )

            logger.warning(
                f"The total number of chunks exceeds {self.config.max_dataset_size}. Increasing the hop size to {self.effective_hop_size_samples // self.config.fs} from {self.config.hop_size_seconds}."
            )

        # Now, build the final chunk map using the effective hop size
        self.chunk_map = []
        for i, duration in enumerate(self.track_durations):
            n_chunks = (
                duration - self.chunk_size_samples
            ) // self.effective_hop_size_samples + 1
            for chunk_idx in range(n_chunks):
                self.chunk_map.append(
                    {
                        "ds_idx": self.track_identifiers[i]["ds_idx"],
                        "track_idx": self.track_identifiers[i]["track_idx"],
                        "chunk_idx": chunk_idx,
                    }
                )

        self.total_size = len(self.chunk_map)

        if self.config.max_dataset_size is not None:
            assert self.total_size <= self.config.max_dataset_size, (
                "Total dataset size exceeds the maximum specified size."
            )

        if self.total_size == 0:
            raise ValueError(
                "Dataset is empty. No tracks were long enough to be chunked."
            )

    def _resolve_index(self, index: int) -> tuple[int, int, int]:
        """
        Resolves a global chunk index to the corresponding datasource, track,
        and chunk index within that track.
        """
        if index >= self.total_size:
            if not self.config.allow_autolooping:
                raise IndexError("Index out of bounds")
            index = index % self.total_size

        chunk_info = self.chunk_map[index]
        return chunk_info["ds_idx"], chunk_info["track_idx"], chunk_info["chunk_idx"]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index: int):
        datasource_index, track_index, chunk_index = self._resolve_index(index)

        track_identifier = self.datasources[datasource_index][track_index]

        item_dict = self._load_audio(track_identifier)

        start_sample = chunk_index * self.effective_hop_size_samples

        chunked_item_dict = self._chunk_and_augment(item_dict, start_sample)

        return chunked_item_dict.model_dump()

    def _chunk_and_augment(
        self,
        item_dict: SourceSeparationItem[np.ndarray],
        start_sample: int,
    ) -> SourceSeparationItem[np.ndarray]:
        """
        Chunks and augments the data deterministically.
        """
        for source in item_dict.sources:
            audio = item_dict.sources[source]["audio"]
            if audio is None or len(audio) == 0:
                item_dict.sources[source]["audio"] = np.zeros(
                    shape=(self.config.n_channels, self.chunk_size_samples),
                    dtype=np.float32,
                )
                continue

            chunked_audio = [
                self._chunk_item(audio_component, start_sample, pad=True)
                for audio_component in audio
            ]

            if self.premix_augmentation is not None:
                chunked_audio = [
                    self.premix_augmentation(
                        chunked_audio_component, sample_rate=self.config.fs
                    )
                    for chunked_audio_component in chunked_audio
                ]

            item_dict.sources[source]["audio"] = sum(chunked_audio)

        mixture = sum(
            item_dict.sources[source]["audio"] for source in item_dict.sources
        )
        item_dict.mixture = {"audio": mixture}

        return item_dict


class RandomChunkSelfQueryDataset(RandomChunkDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = RandomChunkDatasetParams.model_validate(config)
        super().__init__(datasources=datasources, config=config)

    def __getitem__(self, index: int):
        out = super().__getitem__(index)
        target = random.choice(list(out["sources"].keys()))
        out["sources"] = {"target": out["sources"][target]}
        out["queries"] = out["sources"]
        return out


class DeterministicChunkSelfQueryDataset(DeterministicChunkDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = DeterministicChunkDatasetParams.model_validate(config)
        super().__init__(datasources=datasources, config=config)

    def __getitem__(self, index: int):
        out = super().__getitem__(index)
        stems = list(out["sources"].keys())
        target = stems[index % len(stems)]
        out["sources"] = {"target": out["sources"][target]}
        out["queries"] = out["sources"]
        return out


class RandomChunkAutoMixDataset(RandomChunkDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = RandomChunkDatasetParams.model_validate(config)
        super().__init__(datasources=datasources, config=config)

    def __getitem__(self, index: int):
        item_dict = self._load_audio_automix(index)

        chunked_item_dict = self._chunk_and_augment(item_dict, pre_pad=True)

        return chunked_item_dict.model_dump()

    def _load_audio_automix(self, index: int) -> SourceSeparationItem[np.ndarray]:
        datasource = self.datasources[index % len(self.datasources)]
        datasource: MoisesDBStemWiseDatasource

        sources = defaultdict(lambda: {"audio": []})

        n_coarse_stems = random.randint(
            self.config.n_coarse_stems_min, self.config.n_coarse_stems_max
        )
        coarse_stems = datasource.coarse_stems

        p_coarse = np.array(
            [
                datasource.get_coarse_length(coarse_stem=coarse_stem)
                for coarse_stem in coarse_stems
            ]
        )
        p_coarse = p_coarse / p_coarse.sum()

        chosen_coarse_stems = np.random.choice(
            coarse_stems,
            size=min(n_coarse_stems, len(coarse_stems)),
            replace=False,
            p=p_coarse,
        ).tolist()

        for coarse_stem in chosen_coarse_stems:
            n_fine_stems = random.randint(
                self.config.n_fine_stems_min, self.config.n_fine_stems_max
            )
            fine_stems = datasource.fine_stems_by_coarse[coarse_stem]

            p_fine = np.array(
                [
                    datasource.get_fine_length(
                        coarse_stem=coarse_stem, fine_stem=fine_stem
                    )
                    for fine_stem in fine_stems
                ]
            )
            p_fine = p_fine / p_fine.sum()

            chosen_fine_stems = np.random.choice(
                fine_stems,
                size=min(n_fine_stems, len(fine_stems)),
                replace=False,
                p=p_fine,
            )

            for fine_stem in chosen_fine_stems:
                n_fine_tracks = datasource.get_fine_length(
                    coarse_stem=coarse_stem, fine_stem=fine_stem
                )
                # print(coarse_stem, fine_stem, n_fine_tracks)
                fine_track_idx = random.randint(0, n_fine_tracks - 1)
                stem_identifier: StemIdentifier = datasource.get_fine_by_index(
                    coarse_stem=coarse_stem, fine_stem=fine_stem, index=fine_track_idx
                )

                clip_idx = random.randint(0, stem_identifier.n_clips - 1)

                zip = np.load(stem_identifier.full_path, mmap_mode="r")
                clip = zip[f"clip{clip_idx:04d}"]

                if datasource.config.stems is None:
                    composite_key = stem_identifier.name
                else:
                    composite_key = datasource.resolve_stem_to_composite(
                        coarse_stem, fine_stem
                    )
                    # print(f"Resolved {coarse_stem}/{fine_stem} to {composite_key}")

                sources[composite_key]["audio"].append(clip)

        if datasource.config.stems is not None:
            for key in datasource.composite_stems:
                if key not in sources:
                    sources[key] = {"audio": []}

        return SourceSeparationItem(
            mixture=None,
            sources=sources,
            estimates={},
        )


class RandomChunkAutoMixSelfQueryDataset(RandomChunkAutoMixDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = RandomChunkDatasetParams.model_validate(config)
        super().__init__(datasources=datasources, config=config)

        self.query_size_samples = int(self.config.query_size_seconds * self.config.fs)

        self.query_mode = self.config.query_mode

    def __getitem__(self, index: int):
        item_dict = self._load_audio_automix(index)

        chunked_item_dict = self._chunk_and_augment(item_dict, pre_pad=True)
        out = chunked_item_dict.model_dump()

        active_keys = [
            k for k in item_dict.sources if len(item_dict.sources[k]["audio"]) > 0
        ]
        target = random.choice(active_keys)

        if self.query_mode == "same_clip":
            target_clip = item_dict.sources[target]["audio"]

            query_clip = self._chunk_item_random(
                target_clip, chunk_size_samples=self.query_size_samples, pad=True
            )
        elif self.query_mode == "exact":
            query_clip = out["sources"][target]["audio"]
        else:
            raise ValueError(f"Unknown query mode: {self.query_mode}")

        out["sources"] = {"target": out["sources"][target]}
        out["queries"] = {"target": {"audio": query_clip}}

        return out


if __name__ == "__main__":
    config = "/home/hice1/kwatchar3/scratch/banda/configs/data/moisesdb-vdbo-active.yaml"
    import yaml
    import torchaudio as ta
    import torch

    with open(config, "r") as f:
        config = yaml.safe_load(f)["train"]

    ds = RandomChunkAutoMixDataset(
        datasources=[
            MoisesDBStemWiseDatasource(config=config["datasource"][0]["params"])
        ],
        config=config["dataset"]["params"],
    )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        num_workers=16,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2,
    )

    # for batch in dataloader:
    #     pprint(batch)
    #     # break

    for i, item in enumerate(ds):
        # mixture = item["mixture"]["audio"]
        # ta.save(
        #     f"_debug/mixture_{i}.wav",
        #     torch.from_numpy(mixture),
        #     sample_rate=ds.config.fs,
        # )

        for key, source in item["sources"].items():
            print(key, source["audio"].shape)
        #     audio = source["audio"]
        #     ta.save(
        #         f"_debug/source_{i}_{key}.wav",
        #         torch.from_numpy(audio),
        #         sample_rate=ds.config.fs,
        #     )

        if i >= 10:
            break

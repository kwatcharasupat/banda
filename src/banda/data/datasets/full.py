import numpy as np
from banda.data.datasets.base import BaseRegisteredDataset

from banda.data.datasets.base import DatasetParams
from banda.data.datasources.base import BaseRegisteredDatasource


class FullTrackDataset(BaseRegisteredDataset):
    def __init__(
        self, *, datasources: list[BaseRegisteredDatasource], config: DatasetParams
    ):
        config = DatasetParams.model_validate(config)
        super().__init__(datasources=datasources, config=config)
    

    def __getitem__(self, index: int):
        track_identifier = self._get_track_identifier(index)

        item_dict = self._load_audio(track_identifier=track_identifier)
        
        for source in item_dict.sources:
            audio = item_dict.sources[source]["audio"]
            if audio is None or len(audio) == 0:
                item_dict.sources[source]["audio"] = None
                continue

            item_dict.sources[source]["audio"] = sum(audio)

        mixture = sum(
            item_dict.sources[source]["audio"] for source in item_dict.sources
            if item_dict.sources[source]["audio"] is not None
        )
        item_dict.mixture = {"audio": mixture}

        n_samples = mixture.shape[-1]

        for source in item_dict.sources:
            audio = item_dict.sources[source]["audio"]
            if audio is None:
                item_dict.sources[source]["audio"] = np.zeros(
                    shape=(self.config.n_channels, n_samples),
                    dtype=np.float32,
                )

        item_dict.n_samples = n_samples
        item_dict.full_path = track_identifier.full_path

        return item_dict.model_dump()


    def _cache_sizes(self):
        pass

    def __len__(self):
        return self.total_size
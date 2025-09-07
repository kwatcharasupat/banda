from typing import Any, List
from omegaconf import DictConfig
import pytorch_lightning as pl

from banda.data.datasources.base import DatasourceConfig, DatasourceRegistry
from banda.utils import BaseConfig
from torch.utils.data import DataLoader

from .datasets.base import DatasetConfig, DatasetRegistry

import structlog

logger = structlog.get_logger(__name__)


class DataLoaderConfig(BaseConfig):
    batch_size: int
    shuffle: bool
    num_workers: int | None = None
    pin_memory: bool = True
    drop_last: bool = False


class SplitConfig(BaseConfig):
    datasource: List[DatasourceConfig]
    dataset: DatasetConfig
    dataloader: DataLoaderConfig


class DataConfig(BaseConfig):
    train: SplitConfig | None = None
    val: SplitConfig | None = None
    test: SplitConfig | None = None
    predict: SplitConfig | None = None


class SourceSeparationDataModule(pl.LightningDataModule):
    def __init__(self, *, config: DictConfig):
        super().__init__()

        self.config = DataConfig.model_validate(config)

    def train_dataloader(self):
        return self._get_dataloader(config=self.config.train)

    def val_dataloader(self):
        return self._get_dataloader(config=self.config.val)

    def test_dataloader(self):
        return self._get_dataloader(config=self.config.test)

    def predict_dataloader(self):
        return self._get_dataloader(config=self.config.predict)

    def _get_dataloader(self, *, config: SplitConfig):
        logger.info("Creating dataloader with config: %s", config)

        datasources = [
            self._get_datasource(datasource) for datasource in config.datasource
        ]

        dataset = self._get_dataset(config=config.dataset, datasources=datasources)

        dataloader_params = config.dataloader.model_dump()
        dataloader = DataLoader(dataset=dataset, **dataloader_params)

        return dataloader

    def _get_datasource(self, config: DatasourceConfig) -> Any:
        cls_str = config.cls
        params = config.params.model_dump()

        cls = DatasourceRegistry.get_registry().get(cls_str, None)

        if cls is None:
            msg = f"Unknown datasource class: {cls_str}. \n Allowed classes are: {list(DatasourceRegistry.get_registry().keys())}"
            raise ValueError(msg)

        datasource = cls(config=params)

        return datasource

    def _get_dataset(self, config: DatasetConfig, datasources: List[Any]) -> Any:
        cls_str = config.cls
        params = config.params.model_dump()

        cls = DatasetRegistry.get_registry().get(cls_str, None)

        if cls is None:
            msg = f"Unknown dataset class: {cls_str}. \n Allowed classes are: {list(DatasetRegistry.get_registry().keys())}"
            raise ValueError(msg)

        dataset = cls(datasources=datasources, config=params)

        return dataset

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if (
            self.trainer.training
            or self.trainer.validating
            or self.trainer.sanity_checking
        ):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        # do not transfer to device during testing or predicting
        return batch

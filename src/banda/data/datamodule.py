#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pydantic import BaseModel, Field, ConfigDict

import structlog
logger = structlog.get_logger(__name__)

class DatasetConnectorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_")
    config: Dict[str, Any] # This will hold data_root and other connector-specific configs

class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_")
    split: str
    dataset_connector: DatasetConnectorConfig
    fs: int
    premix_transform: Optional[Dict[str, Any]] = None # Changed to Dict[str, Any]
    postmix_transform: Optional[Dict[str, Any]] = None # Changed to Dict[str, Any]

class DataLoaderConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    batch_size: int
    shuffle: bool
    num_workers: Optional[int] = None
    pin_memory: bool = True
    drop_last: bool = False

class DataSplitConfig(BaseModel):
    dataloader: DataLoaderConfig
    dataset: DatasetConfig

class DataModuleConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    train: Optional[DataSplitConfig] = None
    val: Optional[DataSplitConfig] = None
    test: Optional[DataSplitConfig] = None
    predict: Optional[DataSplitConfig] = None

class BaseDataset(Dataset):
    def __init__(self, config: DataSplitConfig):
        self.config = config
        # These fields are now accessed via self.config.dataset_connector.config
        # self.dataset_dir = config.dataset_dir
        self.sample_rate = config.fs # Use fs from DatasetConfig
        # self.segment_duration = config.segment_duration
        # self.use_augmentation = config.use_augmentation
        # self.augmentation_config = config.augmentation_config

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class SourceSeparationDataModule(pl.LightningDataModule): # Renamed BaseDataModule to SourceSeparationDataModule
    def __init__(self, config: DataModuleConfig): # Changed to DataModuleConfig
        super().__init__()
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

        if self.config.train:
            self.train_dataset = self._create_dataset(self.config.train)

        if self.config.val:
            self.val_dataset = self._create_dataset(self.config.val)

        if self.config.test:
            self.test_dataset = self._create_dataset(self.config.test)

        if self.config.predict:
            self.predict_dataset = self._create_dataset(self.config.predict)

    def _create_dataset(self, datasplit_config: DataSplitConfig) -> Dataset:
        """
        Helper method to create a dataset based on the specific dataset config.
        This should be implemented by subclasses.
        """
        # Instantiate the dataset class using its _target_ and config
        
        dataset_config = datasplit_config.dataset
        
        dataset_cls = hydra.utils.get_class(dataset_config.target_)

        return dataset_cls.from_config(config=dataset_config) # Pass the entire DatasetConfig

    def train_dataloader(self):
        if not self.train_dataset:
            return [] # Return empty list if no train dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.dataloader.batch_size,
            shuffle=self.config.train.dataloader.shuffle,
            num_workers=self.config.train.dataloader.num_workers,
            pin_memory=self.config.train.dataloader.pin_memory,
            drop_last=self.config.train.dataloader.drop_last,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return [] # Return empty list if no val dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val.dataloader.batch_size,
            shuffle=False,  # Validation data should not be shuffled
            num_workers=self.config.val.dataloader.num_workers,
            pin_memory=self.config.val.dataloader.pin_memory,
            drop_last=self.config.val.dataloader.drop_last,
        )

    def test_dataloader(self):
        if not self.test_dataset:
            return [] # Return empty list if no test dataset
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test.dataloader.batch_size,
            shuffle=False,  # Test data should not be shuffled
            num_workers=self.config.test.dataloader.num_workers,
            pin_memory=self.config.test.dataloader.pin_memory,
            drop_last=self.config.test.dataloader.drop_last,
        )

    def predict_dataloader(self):
        if not self.predict_dataset:
            return [] # Return empty list if no predict dataset
        return DataLoader(
            self.predict_dataset,
            batch_size=self.config.predict.dataloader.batch_size, # Use common batch_size from DataModuleConfig
            shuffle=False,
            num_workers=self.config.predict.dataloader.num_workers, # Use common num_workers from DataModuleConfig
            pin_memory=self.config.predict.dataloader.pin_memory, # Use common pin_memory from DataModuleConfig
            drop_last=self.config.predict.dataloader.drop_last, # Use common drop_last from DataModuleConfig
        )

    @classmethod
    def from_config(cls, config: DataModuleConfig) -> "SourceSeparationDataModule": # Changed to DataModuleConfig
        """
        Instantiates a DataModule from a DataModuleConfig.
        This method should be overridden by subclasses to return the specific DataModule.
        """
        return cls(config)

# No model_rebuild() calls needed here
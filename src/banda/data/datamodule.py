#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from pydantic import BaseModel, Field, ConfigDict

# from banda.utils.registry import DATASETS_REGISTRY # Removed as per previous instructions

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

class DataModuleConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    train_dataset_config: Optional[DatasetConfig] = None
    val_dataset_config: Optional[DatasetConfig] = None
    test_dataset_config: Optional[DatasetConfig] = None
    predict_dataset_config: Optional[DatasetConfig] = None
    # DataLoader parameters
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    drop_last: bool

class BaseDataset(Dataset):
    def __init__(self, config: DatasetConfig):
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

class SourceSeparationDataModule(L.LightningDataModule): # Renamed BaseDataModule to SourceSeparationDataModule
    def __init__(self, config: DataModuleConfig): # Changed to DataModuleConfig
        super().__init__()
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Download data if needed. This stage is called on 1 GPU/TPU in distributed
        training, it's not called on every device.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data, split datasets, etc. This stage is called on every device.
        """
        if stage == "fit" or stage is None:
            if self.config.train_dataset_config:
                self.train_dataset = self._create_dataset(self.config.train_dataset_config)
            if self.config.val_dataset_config:
                self.val_dataset = self._create_dataset(self.config.val_dataset_config)
        if stage == "test" or stage is None:
            if self.config.test_dataset_config:
                self.test_dataset = self._create_dataset(self.config.test_dataset_config)
            elif self.config.val_dataset_config: # Fallback to val if test is not provided
                self.test_dataset = self._create_dataset(self.config.val_dataset_config)
        if stage == "predict" or stage is None:
            if self.config.predict_dataset_config:
                self.predict_dataset = self._create_dataset(self.config.predict_dataset_config)
            elif self.config.val_dataset_config: # Fallback to val if predict is not provided
                self.predict_dataset = self._create_dataset(self.config.val_dataset_config)

    def _create_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        """
        Helper method to create a dataset based on the specific dataset config.
        This should be implemented by subclasses.
        """
        # Instantiate the dataset class using its _target_ and config
        dataset_cls = hydra.utils.get_class(dataset_config.target_)
        return dataset_cls(config=dataset_config) # Pass the entire DatasetConfig

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not self.train_dataset:
            return [] # Return empty list if no train dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not self.val_dataset:
            return [] # Return empty list if no val dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Validation data should not be shuffled
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not self.test_dataset:
            return [] # Return empty list if no test dataset
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size, # Use common batch_size from DataModuleConfig
            shuffle=False,  # Test data should not be shuffled
            num_workers=self.config.num_workers, # Use common num_workers from DataModuleConfig
            pin_memory=self.config.pin_memory, # Use common pin_memory from DataModuleConfig
            drop_last=self.config.drop_last, # Use common drop_last from DataModuleConfig
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if not self.predict_dataset:
            return [] # Return empty list if no predict dataset
        return DataLoader(
            self.predict_dataset,
            batch_size=self.config.batch_size, # Use common batch_size from DataModuleConfig
            shuffle=False,
            num_workers=self.config.num_workers, # Use common num_workers from DataModuleConfig
            pin_memory=self.config.pin_memory, # Use common pin_memory from DataModuleConfig
            drop_last=self.config.drop_last, # Use common drop_last from DataModuleConfig
        )

    @classmethod
    def from_config(cls, config: DataModuleConfig) -> "SourceSeparationDataModule": # Changed to DataModuleConfig
        """
        Instantiates a DataModule from a DataModuleConfig.
        This method should be overridden by subclasses to return the specific DataModule.
        """
        return cls(config)

# No model_rebuild() calls needed here
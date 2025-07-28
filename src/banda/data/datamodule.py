#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from banda.data.datasets.musdb18hq import SourceSeparationDataset, DatasetConfig
from banda.data.batch_types import FixedStemSeparationBatch
from pydantic import BaseModel, Field


class SourceSeparationDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for source separation datasets.
    """

    def __init__(
        self,
        train_dataset_config: DatasetConfig,
        val_dataset_config: DatasetConfig,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        """
        Args:
            train_dataset_config (DatasetConfig): Configuration for the training dataset.
            val_dataset_config (DatasetConfig): Configuration for the validation dataset.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        """
        super().__init__()
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        """
        Loads data for the given stage.
        """
        if stage == "fit":
            self.train_dataset = SourceSeparationDataset.from_config(OmegaConf.create(self.train_dataset_config.config))
            self.val_dataset = SourceSeparationDataset.from_config(OmegaConf.create(self.val_dataset_config.config))
        elif stage == "validate":
            self.val_dataset = SourceSeparationDataset.from_config(OmegaConf.create(self.val_dataset_config.config))
        elif stage == "test":
            # Implement test dataset loading if needed
            raise NotImplementedError("Test dataset not implemented yet.")
        elif stage == "predict":
            # Implement predict dataset loading if needed
            raise NotImplementedError("Predict dataset not implemented yet.")

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False, # Explicitly set to False
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False, # Explicitly set to False
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch_list: list) -> FixedStemSeparationBatch:
        """
        Custom collate function to convert a list of samples into a Pydantic batch.
        This example assumes FixedStemSeparationBatch.
        For other batch types, this function would need to be more complex or
        multiple collate_fns would be needed.
        """
        # Assuming each item in batch_list is a dictionary with 'audio' and 'identifier'
        # and 'audio' is a TorchInputAudioDict
        
        # Stack tensors for audio.mixture and audio.sources
        # This is a simplified collate_fn. A robust one would handle padding,
        # different source sets, etc.
        
        # Collect all mixtures and sources
        mixtures = [item.audio.mixture for item in batch_list if item.audio.mixture is not None]
        
        # Collect sources for each stem
        all_source_names = set()
        for item in batch_list:
            all_source_names.update(item.audio.sources.keys())

        collated_sources = {name: [] for name in all_source_names}
        for item in batch_list:
            for name in all_source_names:
                if name in item.audio.sources:
                    collated_sources[name].append(item.audio.sources[name])
                else:
                    # Handle missing sources if necessary, e.g., by padding with zeros
                    # For now, assuming all sources are present or will be handled by model
                    pass

        # Stack mixtures and sources
        collated_mixture = torch.stack(mixtures) if mixtures else None
        collated_sources_stacked = {name: torch.stack(tensors) for name, tensors in collated_sources.items() if tensors}

        # Identifiers are typically not stacked as tensors, but kept as a list or processed
        # For simplicity, we'll just take the first identifier's structure and assume batching
        # of identifiers is not directly needed for the Pydantic model.
        # A more complex collate_fn might return a list of identifiers or a batched Identifier model.
        
        # For now, we'll just pass the first identifier's structure.
        # This needs to be refined for proper batching of identifiers.
        # For a real scenario, you might pass a list of identifiers or a batched Identifier model.
        collated_identifier = batch_list[0].identifier # This is a simplification

        return FixedStemSeparationBatch(
            audio=batch_list[0].audio.__class__(
                mixture=collated_mixture,
                sources=collated_sources_stacked
            ),
            identifier=collated_identifier
        )

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Instantiates a SourceSeparationDataModule from a DictConfig.
        """
        # Extract parameters from the config
        train_dataset_config = DatasetConfig(**config.train_dataset_config)
        val_dataset_config = DatasetConfig(**config.val_dataset_config)
        batch_size = config.batch_size
        num_workers = config.get("num_workers", 0)
        pin_memory = config.get("pin_memory", False)

        # Instantiate the DataModule
        return cls(
            train_dataset_config=train_dataset_config,
            val_dataset_config=val_dataset_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class DataModuleConfig(BaseModel): # Inherit directly from BaseModel
    model_config = {'arbitrary_types_allowed': True}
    target_: str = Field(...) # Use target_ instead of _target_
    config: Dict[str, Any] = Field({})
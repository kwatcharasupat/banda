#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.


import torch
import structlog

logger = structlog.get_logger(__name__)

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Set, List
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from banda.data.datasets.base import SourceSeparationDataset, DatasetConnectorConfig
from banda.data.batch_types import FixedStemSeparationBatch, TorchInputAudioDict
from pydantic import BaseModel, Field
from banda.utils.registry import DATASETS_REGISTRY


class SourceSeparationDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for source separation datasets.

    This DataModule handles the setup and loading of training, validation,
    and test datasets, providing DataLoaders for each stage. It also includes
    a custom collate function for handling `FixedStemSeparationBatch` objects.
    """

    def __init__(
        self,
        train_dataset_config: DatasetConnectorConfig,
        val_dataset_config: DatasetConnectorConfig,
        batch_size: int,
        test_dataset_config: Optional[DatasetConnectorConfig] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SourceSeparationDataModule.

        Args:
            train_dataset_config (DatasetConnectorConfig): Configuration for the training dataset.
            val_dataset_config (DatasetConnectorConfig): Configuration for the validation dataset.
            batch_size (int): Batch size for data loaders.
            test_dataset_config (Optional[DatasetConnectorConfig]): Optional configuration for the test dataset.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
            **kwargs: Additional keyword arguments (e.g., for future extensibility).
        """
        super().__init__()
        self.train_dataset_config: DatasetConnectorConfig = train_dataset_config
        self.val_dataset_config: DatasetConnectorConfig = val_dataset_config
        self.test_dataset_config: Optional[DatasetConnectorConfig] = test_dataset_config
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory

        self.train_dataset: Optional[SourceSeparationDataset] = None
        self.val_dataset: Optional[SourceSeparationDataset] = None
        self.test_dataset: Optional[SourceSeparationDataset] = None

    def setup(self, stage: str) -> None:
        """
        Loads data for the given stage (fit, validate, test, predict).

        Args:
            stage (str): The current stage ("fit", "validate", "test", "predict").

        Raises:
            ValueError: If test dataset configuration is not provided for the "test" stage.
            NotImplementedError: If the "predict" stage is requested (not yet implemented).
        """
        logger.info(f"DATASETS_REGISTRY keys: {DATASETS_REGISTRY._data.keys()}")
        if stage == "fit":
            train_dataset_name = self.train_dataset_config.target_.split('.')[-2]
            self.train_dataset = DATASETS_REGISTRY.get(train_dataset_name).from_config(OmegaConf.create(self.train_dataset_config.config))
            val_dataset_name = self.val_dataset_config.target_.split('.')[-2]
            self.val_dataset = DATASETS_REGISTRY.get(val_dataset_name).from_config(OmegaConf.create(self.val_dataset_config.config))
        elif stage == "validate":
            val_dataset_name = self.val_dataset_config.target_.split('.')[-2]
            self.val_dataset = DATASETS_REGISTRY.get(val_dataset_name).from_config(OmegaConf.create(self.val_dataset_config.config))
        elif stage == "test":
            if self.test_dataset_config:
                test_dataset_name = self.test_dataset_config.target_.split('.')[-2]
                self.test_dataset = DATASETS_REGISTRY.get(test_dataset_name).from_config(OmegaConf.create(self.test_dataset_config.config))
            else:
                raise ValueError("Test dataset configuration not provided.")
        elif stage == "predict":
            # Implement predict dataset loading if needed
            raise NotImplementedError("Predict dataset not implemented yet.")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training set.

        Returns:
            DataLoader: A DataLoader instance for the training dataset.
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation set.

        Returns:
            DataLoader: A DataLoader instance for the validation dataset.
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not set up. Call setup('fit') or setup('validate') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test set.

        Note: For testing, the batch size is fixed to 1 as full-length audio
        inference is typically performed.

        Returns:
            DataLoader: A DataLoader instance for the test dataset.
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0, # Typically 0 workers for test to avoid issues with large files
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """
        Transfers the batch to the specified device.

        This method is overridden to handle custom batch types like `FixedStemSeparationBatch`.

        Args:
            batch (Any): The batch to transfer.
            device (torch.device): The target device.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            Any: The batch transferred to the specified device.
        """
        if isinstance(batch, FixedStemSeparationBatch):
            return batch.to_device(device)
        else:
            # Fallback to default behavior for other batch types
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
    def on_exception(self, exception: BaseException) -> None:
        """
        Called when an exception occurs during training, validation, or testing.
        """
        logger.error(f"An exception occurred in DataModule: {exception}")
        raise exception
    
    
    

    def _collate_fn(self, batch_list: list) -> FixedStemSeparationBatch:
        """
        Custom collate function to convert a list of samples into a Pydantic batch.

        This function is designed for `FixedStemSeparationBatch`. For other batch types,
        a more complex or separate collate functions would be needed.

        Args:
            batch_list (list): A list of individual samples (dictionaries) from the dataset.

        Returns:
            FixedStemSeparationBatch: A collated batch of data.
        """
        # Collect all mixtures and sources
        mixtures: List[torch.Tensor] = [item.audio.mixture for item in batch_list if item.audio.mixture is not None]
        
        # Collect sources for each stem
        all_source_names: Set[str] = set()
        for item in batch_list:
            if item.audio.sources:
                all_source_names.update(item.audio.sources.keys())

        collated_sources: Dict[str, List[torch.Tensor]] = {name: [] for name in all_source_names}
        for item in batch_list:
            for name in all_source_names:
                if item.audio.sources and name in item.audio.sources:
                    collated_sources[name].append(item.audio.sources[name])
                else:
                    # If a source is missing for a particular item, handle it (e.g., by padding or skipping)
                    # For now, we assume all sources are present or handled by the dataset.
                    pass

        # Stack mixtures and sources
        collated_mixture: Optional[torch.Tensor] = torch.stack(mixtures) if mixtures else None # Shape: (batch_size, channels, samples)
        collated_sources_stacked: Dict[str, torch.Tensor] = {name: torch.stack(tensors) for name, tensors in collated_sources.items() if tensors} # Shape: Dict[source_name, (batch_size, channels, samples)]

        collated_identifier: Any = batch_list[0].metadata['identifier']
        
        # Generate a dummy query for Banquet model
        # Assuming query_features is 128 as per train_bandit_musdb18hq_test.yaml
        # And batch_size is available from collated_mixture.shape[0]
        query_features = 128 # This should ideally come from model config
        batch_size = collated_mixture.shape[0] if collated_mixture is not None else 1 # Fallback if no mixture
        dummy_query = torch.ones(batch_size, query_features) # Example: tensor of ones

        return FixedStemSeparationBatch(
            mixture=collated_mixture,
            sources=collated_sources_stacked,
            metadata={"identifier": collated_identifier},
            query=dummy_query # Add dummy query
        )

    @classmethod
    def from_config(cls, config: DictConfig) -> "SourceSeparationDataModule":
        """
        Instantiates a SourceSeparationDataModule from a DictConfig.

        Args:
            config (DictConfig): A DictConfig object containing the data module configuration.

        Returns:
            SourceSeparationDataModule: An instance of the SourceSeparationDataModule.
        """
        train_dataset_config: DatasetConnectorConfig = DatasetConnectorConfig(**config.train_dataset_config)
        val_dataset_config: DatasetConnectorConfig = DatasetConnectorConfig(**config.val_dataset_config)
        batch_size: int = config.batch_size
        test_dataset_config: Optional[DatasetConnectorConfig] = DatasetConnectorConfig(**config.test_dataset_config) if "test_dataset_config" in config else None
        num_workers: int = config.get("num_workers", 0)
        pin_memory: bool = config.get("pin_memory", False)

        return cls(
            train_dataset_config=train_dataset_config,
            val_dataset_config=val_dataset_config,
            test_dataset_config=test_dataset_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class DataModuleConfig(BaseModel):
    """
    Pydantic model for data module configuration.

    Attributes:
        name (str): The name of the data module to register.
        config (Dict[str, Any]): A dictionary of configuration parameters for the data module.
    """
    model_config = {'arbitrary_types_allowed': True}
    name: str = Field(...)
    config: Dict[str, Any] = Field({})
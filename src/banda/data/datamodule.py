import os
from typing import Any, Dict, Iterable, List, Optional, Type, Union
from lightning_utilities import apply_to_collection
from pydantic import BaseModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, IterableDataset


from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t

from torch.utils.data.sampler import (
    Sampler,
)


class DataLoaderConfig(BaseModel):
    """
    DataLoaderConfig is a configuration model for defining parameters used in a data loader.

    Attributes:
        batch_size (Optional[int]): Number of samples per batch to load. Defaults to None.
        shuffle (Optional[bool]): Whether to shuffle the data at every epoch. Defaults to None.
        sampler (Union[Sampler, Iterable, None]): Defines the strategy to draw samples from the dataset. Defaults to None.
        batch_sampler (Union[Sampler[List], Iterable[List], None]): Like sampler, but returns a batch of indices at a time. Defaults to None.
        num_workers (int): Number of subprocesses to use for data loading. Defaults to half of the available CPU cores.
        collate_fn (Optional[_collate_fn_t]): Function to merge a list of samples into a batch. Defaults to None.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
        drop_last (bool): If True, drops the last incomplete batch if the dataset size is not divisible by the batch size. Defaults to False.
        timeout (float): Timeout value for collecting a batch from workers. Defaults to 0.
        worker_init_fn (Optional[_worker_init_fn_t]): Function to initialize each worker process. Defaults to None.
        multiprocessing_context: Context for multiprocessing. Defaults to None.
        generator: Generator used for random number generation. Defaults to None.
        prefetch_factor (Optional[int]): Number of batches loaded in advance by each worker. Defaults to None.
        persistent_workers (bool): If True, the data loader will not shut down the worker processes after a dataset has been consumed. Defaults to False.
        pin_memory_device (str): Device for pin memory. Defaults to an empty string.
        in_order (bool): If True, ensures data is loaded in order. Defaults to True.
    """

    batch_size: Optional[int]
    shuffle: Optional[bool] = None
    sampler: Union[Sampler, Iterable, None] = None
    batch_sampler: Union[Sampler[List], Iterable[List], None] = None
    num_workers: int = (os.cpu_count() or 0) // 2
    collate_fn: Optional[_collate_fn_t] = None
    pin_memory: bool = True
    drop_last: bool = False
    timeout: float = 0
    worker_init_fn: Optional[_worker_init_fn_t] = None
    multiprocessing_context = None
    generator = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""
    in_order: bool = True


class DatasetConfig(BaseModel):
    """
    DatasetConfig is a configuration model for defining parameters used in a dataset.

    Attributes:
        _target_ (str): Specifies the target class or function to be used.
        params (Dict[str, Any]): Parameters for initializing the dataset.
    """

    _target_: str
    dataset_params: Dict[str, Any] = {}


class SplitConfig(BaseModel):
    """
    SplitConfig is a configuration model that defines the structure for dataset and dataloader configurations.

    Attributes:
        dataset (DatasetConfig): Configuration settings for the dataset.
        dataloader (DataLoaderConfig): Configuration settings for the dataloader.
    """

    dataset: DatasetConfig
    dataloader: DataLoaderConfig


class SplitConfigs(BaseModel):
    """
    SplitConfigs is a data model that defines optional configurations for different data splits
    used in a machine learning workflow. Each split (train, val, test, predict) can have its
    own configuration specified using the SplitConfig type.

    Attributes:
        train (Optional[SplitConfig]): Configuration for the training data split. Defaults to None.
        val (Optional[SplitConfig]): Configuration for the validation data split. Defaults to None.
        test (Optional[SplitConfig]): Configuration for the testing data split. Defaults to None.
        predict (Optional[SplitConfig]): Configuration for the prediction data split. Defaults to None.
    """

    train: Optional[SplitConfig] = None
    val: Optional[SplitConfig] = None
    test: Optional[SplitConfig] = None
    predict: Optional[SplitConfig] = None


class DataConfig(BaseModel):
    """
    DataConfig is a configuration model for data-related settings.

    Attributes:
        _target_ (Optional[str]): Specifies the target class or function to be used. Defaults to None.
        split_dataloaders (SplitConfigs): Configuration for splitting data into different dataloaders.
        kwargs (Dict[str, Any]): Additional keyword arguments for customization.
    """

    _target_: Optional[str] = None
    split_dataloaders: SplitConfigs
    datamodule_params: Dict[str, Any] = {}


def instantiate_dataset(config: DatasetConfig) -> Dataset:
    """
    Instantiates and returns a dataset object based on the provided configuration.

    Args:
        config (DatasetConfig): A configuration object containing the target dataset class
                                (specified as a string in the `_target_` attribute) and
                                initialization parameters (in the `params` attribute).

    Returns:
        Dataset: An instance of the dataset class specified in the configuration.

    Raises:
        NameError: If the `_target_` attribute in the configuration does not correspond
                   to a valid class name.
        TypeError: If the parameters in `params` do not match the expected arguments
                   for the dataset class constructor.
    """
    dataset_cls: Type[Dataset] = eval(config._target_)
    dataset = dataset_cls(**config.dataset_params)
    return dataset


def instantiate_datamodule(
    config: DataConfig,
) -> pl.LightningDataModule:
    """
    Instantiates a PyTorch Lightning DataModule based on the provided configuration.

    Args:
        config (DataConfig): The configuration object containing the details for
            creating the DataModule, including the target class and its arguments.

    Returns:
        pl.LightningDataModule: An instance of the LightningDataModule configured
            with the specified datasets and dataloaders.

    Raises:
        AttributeError: If the `config._target_` is invalid or the specified class
            cannot be instantiated.

    Notes:
        - The function dynamically evaluates the target class specified in the
          `config._target_` attribute. If no target is provided, it defaults to
          `pl.LightningDataModule`.
        - Datasets and dataloaders are instantiated for each split (e.g., train, val, test)
          based on the configuration. The `shuffle` attribute for the dataloader is
          automatically set to `True` for the training split unless overridden.
        - Supports handling of `IterableDataset` by disabling shuffling when necessary.

    Adapted from:
        - https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/datamodule.html#LightningDataModule.from_datasets
    """
    if config._target_ is None:
        dm_cls: Type[pl.LightningDataModule] = pl.LightningDataModule
    else:
        dm_cls: Type[pl.LightningDataModule] = eval(config._target_)

    datamodule = dm_cls(**config.datamodule_params)  # type: ignore

    for split, split_config in config.model_dump().items():
        if split_config is None:
            continue

        dataset = instantiate_dataset(split_config.dataset)

        def dataloader_(dataset: Dataset) -> DataLoader:
            if split_config.dataloader.shuffle is None:
                shuffle = split == "train"
            shuffle = shuffle and not isinstance(dataset, IterableDataset)

            split_config.dataloader.shuffle = shuffle

            return DataLoader(
                dataset,
                **split_config.dataloader.model_dump(),
            )

        split_dataloader = apply_to_collection(dataset, Dataset, dataloader_)

        setattr(datamodule, f"{split}_dataloader", split_dataloader)  # type: ignore[method-assign]
    return datamodule

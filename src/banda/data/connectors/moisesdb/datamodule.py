from typing import Mapping, Optional
import warnings

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bandx.data.datasets.base import from_datasets
from bandx.data.moisesdb.datasets.precomputed_dual_queries import (
    MoisesDBDeterministicChunkPrecomputedDeterministicPairedQueryDataset,
    MoisesDBRandomChunkPrecomputedRandomPairedQueryDataset,
)
from bandx.data.moisesdb.datasets.precomputed_query import (
    MoisesDBDeterministicChunkPrecomputedDeterministicSingleQueryDataset,
    MoisesDBRandomChunkPrecomputedRandomSingleQueryDataset,
)
from bandx.data.moisesdb.datasets.query import (
    MoisesDBRandomChunkDualRandomQueryDataset,
    MoisesDBRandomChunkRandomQueryDataset,
)
from bandx.data.moisesdb.datasets.raw import (
    MoisesDBDeterministicChunkedRawStemsDataset,
    MoisesDBRandomChunkedRawStemsDataset,
)


def _get_dataset_cls(precomputed_query: bool, dual_query: bool, is_test: bool = False):
    """Helper function to determine the dataset class based on query flags."""
    if precomputed_query and dual_query:
        return (
            MoisesDBDeterministicChunkPrecomputedDeterministicPairedQueryDataset
            if not is_test
            else MoisesDBRandomChunkPrecomputedRandomPairedQueryDataset
        )
    if precomputed_query:
        return (
            MoisesDBDeterministicChunkPrecomputedDeterministicSingleQueryDataset
            if not is_test
            else MoisesDBRandomChunkPrecomputedRandomSingleQueryDataset
        )
    if dual_query:
        if is_test:
            warnings.warn(
                "random dual_query shouldn't actually be used for test dataset"
            )
        return MoisesDBRandomChunkDualRandomQueryDataset
    return MoisesDBRandomChunkRandomQueryDataset


def MoisesDBDefaultDataModule(
    data_root: str,
    batch_size: int,
    num_workers: Optional[int] = None,
    train_kwargs: Optional[Mapping] = None,
    val_kwargs: Optional[Mapping] = None,
    test_kwargs: Optional[Mapping] = None,
    datamodule_kwargs: Optional[Mapping] = None,
) -> pl.LightningDataModule:
    train_kwargs = train_kwargs or {}
    val_kwargs = val_kwargs or {}
    test_kwargs = test_kwargs or {}
    datamodule_kwargs = datamodule_kwargs or {}

    train_dataset = _get_dataset_cls(
        train_kwargs.pop("precomputed_query", False),
        train_kwargs.pop("dual_query", False),
    )(data_root=data_root, split="train", **train_kwargs)

    val_dataset = _get_dataset_cls(
        val_kwargs.pop("precomputed_query", False),
        val_kwargs.pop("dual_query", False),
    )(data_root=data_root, split="val", **val_kwargs)

    test_dataset = _get_dataset_cls(
        test_kwargs.pop("precomputed_query", False),
        test_kwargs.pop("dual_query", False),
        is_test=True,
    )(data_root=data_root, split="test", **test_kwargs)

    datamodule = from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **datamodule_kwargs,
    )
    datamodule.predict_dataloader = datamodule.test_dataloader  # type: ignore[method-assign]
    return datamodule


def _create_dataloader(
    dataset, batch_size, num_workers, shuffle=False, drop_last=False
):
    """Helper function to create a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # collate_fn=raw_collate_fn, #FIXME: this is not implemented yet
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class MoisesRawDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int = 8,
        train_kwargs: Optional[Mapping] = None,
        val_kwargs: Optional[Mapping] = None,
        test_kwargs: Optional[Mapping] = None,
        datamodule_kwargs: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        train_kwargs = train_kwargs or {}
        val_kwargs = val_kwargs or {}
        test_kwargs = test_kwargs or {}
        datamodule_kwargs = datamodule_kwargs or {}

        train_cls = (
            MoisesDBDeterministicChunkedRawStemsDataset
            if datamodule_kwargs.pop("premake", False)
            else MoisesDBRandomChunkedRawStemsDataset
        )
        rank = datamodule_kwargs.pop("rank", 0)
        n_jobs = datamodule_kwargs.pop("n_jobs", 1)

        self.train_dataset = train_cls(
            data_root=data_root, split="train", **train_kwargs, rank=rank, n_jobs=n_jobs
        )
        self.val_dataset = MoisesDBDeterministicChunkedRawStemsDataset(
            data_root=data_root, split="val", **val_kwargs, rank=rank, n_jobs=n_jobs
        )
        self.test_dataset = MoisesDBDeterministicChunkedRawStemsDataset(
            data_root=data_root, split="test", **test_kwargs, rank=rank, n_jobs=n_jobs
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return _create_dataloader(self.train_dataset, self.batch_size, self.num_workers)

    def val_dataloader(self):
        return _create_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    def test_dataloader(self):
        return _create_dataloader(self.test_dataset, self.batch_size, self.num_workers)

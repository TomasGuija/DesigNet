from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from designet.font_dataset import FontDataset


class FontDataModule(L.LightningDataModule):
    """Lightning wrapper around :class:`designet.font_dataset.FontDataset`."""

    def __init__(
        self,
        data_dir: str | Path,
        max_num_groups: int,
        max_seq_len: int,
        encoding_letters: Sequence[str],
        decoding_letters: Sequence[str],
        max_total_len: int | None = None,
        train_csv_path: str | Path | None = None,
        val_csv_path: str | Path | None = None,
        test_csv_path: str | Path | None = None,
        csv_path: str | Path | None = None,
        split: Sequence[float] = (0.9, 0.05, 0.05),
        seed: int = 42,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.max_num_groups = max_num_groups
        self.max_seq_len = max_seq_len
        self.max_total_len = max_total_len
        self.encoding_letters = list(encoding_letters)
        self.decoding_letters = list(decoding_letters)

        self.train_csv_path = Path(train_csv_path) if train_csv_path is not None else None
        self.val_csv_path = Path(val_csv_path) if val_csv_path is not None else None
        self.test_csv_path = Path(test_csv_path) if test_csv_path is not None else None
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.split = tuple(float(value) for value in split)
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (num_workers > 0) if persistent_workers is None else bool(persistent_workers)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        if self._has_split_csvs:
            self.train_dataset = self._build_dataset(self._require_path(self.train_csv_path, "train_csv_path"))
            self.val_dataset = self._build_dataset(self._require_path(self.val_csv_path, "val_csv_path"))
            self.test_dataset = self._build_dataset(self._require_path(self.test_csv_path, "test_csv_path"))
            return

        if self.csv_path is None:
            raise ValueError("Provide either train/val/test CSV paths or a single csv_path to split.")

        dataset = self._build_dataset(self.csv_path)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            self._split_lengths(len(dataset)),
            generator=torch.Generator().manual_seed(self.seed),
        )

    @property
    def _has_split_csvs(self) -> bool:
        return any(path is not None for path in (self.train_csv_path, self.val_csv_path, self.test_csv_path))

    @staticmethod
    def _require_path(path: Path | None, name: str) -> Path:
        if path is None:
            raise ValueError(f"{name} is required when using explicit dataset splits.")
        return path

    def _build_dataset(self, csv_path: Path) -> FontDataset:
        return FontDataset(
            data_dir=self.data_dir,
            csv_path=csv_path,
            max_num_groups=self.max_num_groups,
            max_seq_len=self.max_seq_len,
            max_total_len=self.max_total_len,
            encoding_letters=self.encoding_letters,
            decoding_letters=self.decoding_letters,
            compute_continuity=True,
            compute_line_alignment=True,
            compute_auxiliary_points=True,
        )

    def _split_lengths(self, dataset_len: int) -> list[int]:
        if len(self.split) != 3:
            raise ValueError(f"split must contain three values, got {self.split}")

        split_sum = sum(self.split)
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"split values must sum to 1.0, got {split_sum}")

        train_len = int(dataset_len * self.split[0])
        val_len = int(dataset_len * self.split[1])
        test_len = dataset_len - train_len - val_len
        return [train_len, val_len, test_len]

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self._require_dataset(self.train_dataset, "train"), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self._require_dataset(self.val_dataset, "validation"), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self._require_dataset(self.test_dataset, "test"), shuffle=False)

    @staticmethod
    def _require_dataset(dataset: Dataset | None, name: str) -> Dataset:
        if dataset is None:
            raise RuntimeError(f"{name} dataset is not initialized. Did you call setup()?")
        return dataset

    def _dataloader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            collate_fn=FontDataset.collate_fn,
        )

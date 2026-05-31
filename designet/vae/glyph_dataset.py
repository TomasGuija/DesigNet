"""
Glyph dataset for VAE workflows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from designet.svg_utils import load_svg_as_tensor_sample
from designet.tensor_utils import PAD_VAL, collate_stack_samples, sequence_length_mask


class GlyphDataset(Dataset):
    """
    Dataset that loads glyphs listed in a metadata CSV file.

    The first CSV column is assumed to contain glyph ids. Each id must be the
    relative path from ``data_dir`` to the SVG file, without the ``.svg`` suffix.
    For example, ``family_1740/font_0001/O_79`` maps to:
    ``data_dir/family_1740/font_0001/O_79.svg``.

    The dataset filters samples using the CSV metadata columns ``nb_groups``,
    ``max_len_group``, and ``total_len``.

    Args:
        data_dir: Root directory containing the SVG glyph files.
        csv_path: Path to the metadata CSV file.
        max_num_groups: Maximum number of SVG path groups allowed by the model.
        max_seq_len: Maximum number of commands per path group, excluding SOS/EOS.
        max_total_len: Optional maximum total number of commands per glyph.
    """

    def __init__(
        self,
        data_dir: str | Path,
        csv_path: str | Path,
        max_num_groups: int,
        max_seq_len: int,
        max_total_len: int | None = None,
        *,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.max_num_groups = max_num_groups
        self.max_seq_len = max_seq_len
        self.max_total_len = max_total_len
        self.use_cache = use_cache

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if self.df.empty:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

        self.id_col = self.df.columns[0]
        self.df = self._filter_df(self.df).reset_index(drop=True)

        if self.df.empty:
            raise ValueError("No glyphs remain after applying dataset filters")

        self.cache_dir = self._resolve_cache_dir(cache_dir) if use_cache else None
        self._memory_cache: dict[int, Dict[str, Any]] = {}

    def _filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = sequence_length_mask(
            df,
            max_num_groups=self.max_num_groups,
            max_seq_len=self.max_seq_len,
            max_total_len=self.max_total_len,
        )
        return df[mask]

    def _resolve_cache_dir(self, cache_dir: str | Path | None) -> Path | None:
        settings = {
            "version": 2,
            "csv_path": str(self.csv_path.resolve()),
            "data_dir": str(self.data_dir.resolve()),
            "ids": self.df[self.id_col].astype(str).tolist(),
            "max_num_groups": self.max_num_groups,
            "max_seq_len": self.max_seq_len,
            "max_total_len": self.max_total_len,
            "center": True,
            "outputs": ["commands", "args", "continuity", "alignment", "aux_points"],
        }
        digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        root = Path(cache_dir) if cache_dir is not None else self.data_dir / ".designet_cache" / "glyph_dataset"
        path = root / digest

        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.warning("Disabling glyph dataset cache; could not create %s: %s", path, exc)
            return None

        return path

    def __len__(self) -> int:
        """Return the number of valid glyphs."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._memory_cache:
            return self._memory_cache[idx]

        row = self.df.iloc[idx]
        glyph_id = str(row[self.id_col])
        cached = self._load_from_cache(idx, glyph_id)
        if cached is not None:
            self._memory_cache[idx] = cached
            return cached

        sample = self._build_sample(glyph_id)
        self._save_to_cache(idx, glyph_id, sample)
        self._memory_cache[idx] = sample
        return sample

    def _build_sample(self, glyph_id: str) -> Dict[str, Any]:
        svg_path = self.data_dir / f"{glyph_id}.svg"

        sample = load_svg_as_tensor_sample(
            svg_path,
            max_num_groups=self.max_num_groups,
            max_seq_len=self.max_seq_len,
            pad_val=PAD_VAL,
            center=True,
            compute_continuity=True,
            compute_line_alignment=True,
            compute_auxiliary_points=True,
            batch=False,
        )

        return {
            "id": glyph_id,
            **sample,
        }

    def _cache_path(self, idx: int, glyph_id: str) -> Path | None:
        if self.cache_dir is None:
            return None

        svg_path = self.data_dir / f"{glyph_id}.svg"
        try:
            stat = svg_path.stat()
            file_state = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            file_state = "missing"

        key = hashlib.sha1(f"{idx}:{glyph_id}:{file_state}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def _load_from_cache(self, idx: int, glyph_id: str) -> Dict[str, Any] | None:
        path = self._cache_path(idx, glyph_id)
        if path is None or not path.exists():
            return None

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logging.warning("Ignoring unreadable glyph cache file %s: %s", path, exc)
            return None

    def _save_to_cache(self, idx: int, glyph_id: str, sample: Dict[str, Any]) -> None:
        path = self._cache_path(idx, glyph_id)
        if path is None:
            return

        try:
            torch.save(sample, path)
        except Exception as exc:
            logging.warning("Could not write glyph cache file %s: %s", path, exc)

    @staticmethod
    def collate_fn(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_stack_samples(
            samples,
            tensor_keys=("commands", "args", "continuity", "alignment", "aux_points"),
            meta_keys=("id",),
        )

"""
Glyph dataset for VAE workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.max_num_groups = max_num_groups
        self.max_seq_len = max_seq_len
        self.max_total_len = max_total_len

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

    def _filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = sequence_length_mask(
            df,
            max_num_groups=self.max_num_groups,
            max_seq_len=self.max_seq_len,
            max_total_len=self.max_total_len,
        )
        return df[mask]

    def __len__(self) -> int:
        """Return the number of valid glyphs."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        glyph_id = str(row[self.id_col])
        svg_path = self.data_dir / f"{glyph_id}.svg"

        sample = load_svg_as_tensor_sample(
            svg_path,
            max_num_groups=self.max_num_groups,
            max_seq_len=self.max_seq_len,
            pad_val=PAD_VAL,
            center=True,
            batch=False,
        )

        return {
            "id": glyph_id,
            **sample,
        }

    @staticmethod
    def collate_fn(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_stack_samples(samples, meta_keys=("id",))

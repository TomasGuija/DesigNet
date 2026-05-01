"""
Font dataset for generative SVG workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
from torch.utils.data import Dataset

from designet.svg_utils import load_svg_as_tensor_sample, to_cp
from designet.tensor_utils import (
    PAD_VAL,
    collate_stack_samples,
    sequence_length_mask,
    stack_font_glyph_samples,
)


class FontDataset(Dataset):
    """
    Dataset that loads full fonts listed in a glyph-level metadata CSV.

    Each item is one font, composed of several glyph SVGs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        csv_path: str | Path,
        max_num_groups: int,
        max_seq_len: int,
        max_total_len: int | None = None,
        encoding_letters: Sequence[str] | None = None,
        decoding_letters: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.max_num_groups = max_num_groups
        self.max_seq_len = max_seq_len
        self.max_total_len = max_total_len

        self.encoding_letters = list(encoding_letters or [])
        self.decoding_letters = list(decoding_letters or [])

        self.encoding_cp = [to_cp(letter) for letter in self.encoding_letters]
        self.decoding_cp = [to_cp(letter) for letter in self.decoding_letters]
        self.required_letters_cps = self.encoding_cp + self.decoding_cp

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if self.df.empty:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

        self.df = self._filter_df(self.df).reset_index(drop=True)

        if self.df.empty:
            raise ValueError("No fonts contain all required glyphs")

        self.font_dirs = sorted(self.df["font_dir"].unique())

    def _filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["id", "uni", "nb_groups", "max_len_group", "total_len"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required metadata columns: {missing}")

        df = df.copy()
        df["uni"] = df["uni"].astype(str)
        df["font_dir"] = df["id"].apply(lambda x: str(Path(x).parent))

        length_mask = sequence_length_mask(
            df,
            max_num_groups=self.max_num_groups,
            max_seq_len=self.max_seq_len,
            max_total_len=self.max_total_len,
        )

        df["fits_limits"] = length_mask

        valid_fonts = []
        required_letters = set(map(str, self.required_letters_cps))

        for font_dir, group in df.groupby("font_dir"):
            available_letters = set(group["uni"])

            has_all_required = required_letters.issubset(available_letters)
            all_required_fit = group["fits_limits"].all()

            if has_all_required and all_required_fit:
                valid_fonts.append(font_dir)

        df = df[df["font_dir"].isin(valid_fonts)]
        df = df.drop(columns=["fits_limits"])

        return df

    def __len__(self) -> int:
        return len(self.font_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        font_dir = self.font_dirs[idx]
        font_df = self.df[self.df["font_dir"] == font_dir].sort_values("id")

        encoding_rows = font_df[font_df["uni"].astype(int).isin(self.encoding_cp)]
        decoding_rows = font_df[font_df["uni"].astype(int).isin(self.decoding_cp)]

        glyph_rows = list(encoding_rows.itertuples(index=False)) + list(decoding_rows.itertuples(index=False))

        glyph_samples = []

        for row in glyph_rows:
            glyph_id = str(row.id)
            svg_path = self.data_dir / f"{glyph_id}.svg"

            glyph_samples.append(
                load_svg_as_tensor_sample(
                    svg_path,
                    max_num_groups=self.max_num_groups,
                    max_seq_len=self.max_seq_len,
                    pad_val=PAD_VAL,
                    center=True,
                    batch=False,
                )
            )

        out = stack_font_glyph_samples(glyph_samples, batch=False)

        return {
            "font_dir": font_dir,
            **out,
        }

    @staticmethod
    def collate_fn(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_stack_samples(samples, meta_keys=("font_dir",))

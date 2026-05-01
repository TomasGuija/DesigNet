from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

from designet.svg_utils import index_svg_paths, load_svg_as_tensor_sample
from designet.tensor_utils import PAD_VAL, collate_cat_samples


@dataclass
class GlyphLoaderConfig:
    max_num_groups: int
    max_seq_len: int
    compute_continuity: bool = False
    compute_line_alignment: bool = False


class GlyphLoader:
    """Loads glyph SVG files and converts them into model-ready tensors."""

    def __init__(self, font_dir: str | Path, cfg: GlyphLoaderConfig):
        self.font_dir = Path(font_dir)
        self.cfg = cfg

        if not self.font_dir.exists():
            raise FileNotFoundError(f"Font directory not found: {self.font_dir}")

        self.name_to_path = index_svg_paths(self.font_dir, require_non_empty=True)

    def available_glyphs(self) -> list[str]:
        return sorted(self.name_to_path.keys())

    def get(self, glyph_name: str) -> dict:
        if glyph_name not in self.name_to_path:
            available = ", ".join(self.available_glyphs())
            raise KeyError(f"Glyph '{glyph_name}' not found in {self.font_dir}. Available: {available}")

        return load_svg_as_tensor_sample(
            self.name_to_path[glyph_name],
            max_num_groups=self.cfg.max_num_groups,
            max_seq_len=self.cfg.max_seq_len,
            pad_val=PAD_VAL,
            center=True,
            compute_continuity=self.cfg.compute_continuity,
            compute_line_alignment=self.cfg.compute_line_alignment,
            batch=True,
        )

    def __getitem__(self, glyph_name: str) -> dict:
        return self.get(glyph_name)

    @staticmethod
    def collate_samples_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_cat_samples(samples)

    def get_batch(
        self,
        glyph_keys: Sequence[str],
        *,
        skip_failed_glyphs: bool = True,
    ) -> Dict[str, Any]:
        samples: list[Dict[str, Any]] = []

        for glyph_key in glyph_keys:
            try:
                sample = self.get(glyph_key)
                samples.append(sample)
            except Exception:
                if not skip_failed_glyphs:
                    raise
                continue

        if not samples:
            raise ValueError("No valid glyph samples could be loaded")

        return self.collate_samples_batch(samples)

    def iter_font_batches(
        self,
        batch_size: int,
        *,
        glyph_keys: Sequence[str] | None = None,
        skip_failed_glyphs: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        keys = list(glyph_keys) if glyph_keys is not None else self.available_glyphs()

        for start in range(0, len(keys), batch_size):
            chunk = keys[start : start + batch_size]
            yield self.get_batch(chunk, skip_failed_glyphs=skip_failed_glyphs)

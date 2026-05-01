from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from designet.svg_utils import index_svg_paths, load_svg_as_tensor_sample
from designet.tensor_utils import PAD_VAL, collate_cat_samples, stack_font_glyph_samples


@dataclass
class FontLoaderConfig:
    """Configuration for converting one font directory into model-ready tensors."""

    max_num_groups: int
    max_seq_len: int
    encoding_letters: Sequence[str]
    decoding_letters: Sequence[str]
    compute_continuity: bool = False
    compute_line_alignment: bool = False


class FontLoader:
    """Load one font folder into tensors expected by `FontConditionalSVGTransformer`."""

    def __init__(self, font_dir: str | Path, cfg: FontLoaderConfig):
        self.font_dir = Path(font_dir)
        self.cfg = cfg

        if not self.font_dir.exists():
            raise FileNotFoundError(f"Font directory not found: {self.font_dir}")

        self.encoding_glyphs = [str(x) for x in self.cfg.encoding_letters]
        self.decoding_glyphs = [str(x) for x in self.cfg.decoding_letters]
        self.ordered_glyphs = self._resolve_ordered_glyphs()

        self.name_to_path = index_svg_paths(self.font_dir)

        missing = [name for name in self.ordered_glyphs if name not in self.name_to_path]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Missing required glyphs in {self.font_dir}: {missing_str}")

    def _resolve_ordered_glyphs(self) -> list[str]:
        """Resolve glyph order to match training-time dataset behavior.

        Training loader logic was: sort SVG files by path, then split into
        encoding and decoding subsets while preserving that sorted order.
        """
        required = self.encoding_glyphs + self.decoding_glyphs

        sorted_stems = [p.stem for p in sorted(self.font_dir.glob("*.svg"))]
        enc_set = set(self.encoding_glyphs)
        dec_set = set(self.decoding_glyphs)

        encoding_sorted = [stem for stem in sorted_stems if stem in enc_set]
        decoding_sorted = [stem for stem in sorted_stems if stem in dec_set]
        ordered = encoding_sorted + decoding_sorted

        missing_required = [name for name in required if name not in ordered]
        if missing_required:
            missing_str = ", ".join(missing_required)
            raise KeyError(f"Missing required glyphs in {self.font_dir}: {missing_str}")

        return ordered

    def _glyph_to_sample(self, svg_path: Path) -> Dict[str, torch.Tensor]:
        return load_svg_as_tensor_sample(
            svg_path,
            max_num_groups=self.cfg.max_num_groups,
            max_seq_len=self.cfg.max_seq_len,
            pad_val=PAD_VAL,
            center=True,
            compute_continuity=self.cfg.compute_continuity,
            compute_line_alignment=self.cfg.compute_line_alignment,
            batch=False,
        )

    def get(self) -> Dict[str, Any]:
        per_glyph = [self._glyph_to_sample(self.name_to_path[name]) for name in self.ordered_glyphs]

        out = stack_font_glyph_samples(
            per_glyph,
            batch=True,
            command_key="input_commands",
            args_key="input_args",
        )

        out["glyph_names"] = self.ordered_glyphs
        out["font_dir"] = self.font_dir

        return out

    @staticmethod
    def collate_fonts(font_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out = collate_cat_samples(
            font_samples,
            tensor_keys=("input_commands", "input_args"),
            optional_keys=("continuity", "alignment"),
        )

        out["glyph_names"] = font_samples[0].get("glyph_names")
        return out

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from designet.difflib.tensor import SVGTensor
from designet.svglib.geom import Bbox, Point
from designet.svglib.svg import SVG
from designet.vae.glyph_loader import (
    PAD_VAL,
    compute_continuity_tensor,
    compute_line_alignment_tensor,
)


@dataclass
class FontLoaderConfig:
    """Configuration for converting one font directory into model-ready tensors."""

    max_num_groups: int
    max_seq_len: int
    encoding_letters: Sequence[str]
    decoding_letters: Sequence[str]
    normalize: bool = False
    center: bool = False
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

        self.name_to_path = self._index_glyph_paths(self.font_dir)

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

    @staticmethod
    def _index_glyph_paths(font_dir: Path) -> Dict[str, Path]:
        """Build a glyph-name -> SVG path index from file stems."""
        name_to_path: Dict[str, Path] = {}

        for p in sorted(font_dir.glob("*.svg")):
            stem = p.stem
            if stem not in name_to_path:
                name_to_path[stem] = p

        return name_to_path

    def _preprocess(self, svg: SVG) -> SVG:
        """Apply optional normalization/centering to match training-time conventions."""
        if self.cfg.normalize:
            bbox = svg.bbox()
            if bbox is not None:
                svg = svg.translate(Point(-bbox.xy.x, -bbox.xy.y))
                w = max(bbox.size.x, 1e-12)
                h = max(bbox.size.y, 1e-12)
                s = 1.0 / max(w, h)
                svg = svg._apply_to_paths("scale", s)
                svg.viewbox = Bbox(1)

        if self.cfg.center:
            bbox = svg.bbox()
            if bbox is not None:
                cx = bbox.xy.x + bbox.size.x / 2
                cy = bbox.xy.y + bbox.size.y / 2
                svg = svg.translate(Point(-cx, -cy))

        return svg

    def _glyph_to_sample(self, svg_path: Path) -> Dict[str, torch.Tensor]:
        """Convert one glyph SVG to grouped command/arg tensors."""
        svg = SVG.load_svg(str(svg_path))
        svg = self._preprocess(svg)

        t_sep, fillings = svg.to_tensor(concat_groups=False, PAD_VAL=PAD_VAL), svg.to_fillings()
        if not t_sep:
            raise ValueError(f"No valid tensor groups found in {svg_path}")

        t_sep = list(t_sep)
        fillings = list(fillings)
        num_args = t_sep[0].shape[1]

        pad_len = max(self.cfg.max_num_groups - len(t_sep), 0)
        t_sep.extend([torch.empty(0, num_args)] * pad_len)
        fillings.extend([0] * pad_len)

        t_sep_tensors = [
            SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(seq_len=self.cfg.max_seq_len + 2)
            for t, f in zip(t_sep, fillings)
        ]

        commands = torch.stack([t.cmds() for t in t_sep_tensors])
        args = torch.stack([t.args(with_start_pos=True) for t in t_sep_tensors])

        out: Dict[str, torch.Tensor] = {
            "commands": commands,
            "args": args,
        }

        if self.cfg.compute_continuity:
            out["continuity"] = torch.stack([compute_continuity_tensor(c, a[..., -6:]) for c, a in zip(commands, args)])

        if self.cfg.compute_line_alignment:
            out["alignment"] = torch.stack(
                [compute_line_alignment_tensor(c, a[..., -6:]) for c, a in zip(commands, args)]
            )

        return out

    def get(self) -> Dict[str, Any]:
        """Return one font sample with batch shape [1, N, ...] for model.forward."""
        per_glyph = [self._glyph_to_sample(self.name_to_path[name]) for name in self.ordered_glyphs]

        commands = torch.stack([g["commands"] for g in per_glyph], dim=0).unsqueeze(0)
        args = torch.stack([g["args"] for g in per_glyph], dim=0).unsqueeze(0)

        out: Dict[str, Any] = {
            "input_commands": commands,
            "input_args": args,
            "glyph_names": self.ordered_glyphs,
            "font_dir": self.font_dir,
        }

        if self.cfg.compute_continuity:
            continuity = torch.stack([g["continuity"] for g in per_glyph], dim=0).unsqueeze(0)
            out["continuity"] = continuity

        if self.cfg.compute_line_alignment:
            alignment = torch.stack([g["alignment"] for g in per_glyph], dim=0).unsqueeze(0)
            out["alignment"] = alignment

        return out

    @staticmethod
    def collate_fonts(font_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Concatenate multiple font samples (already shaped [1, N, ...]) into a batch."""
        if not font_samples:
            raise ValueError("font_samples must contain at least one element")

        out: Dict[str, Any] = {
            "input_commands": torch.cat([s["input_commands"] for s in font_samples], dim=0),
            "input_args": torch.cat([s["input_args"] for s in font_samples], dim=0),
        }

        if all("continuity" in s for s in font_samples):
            out["continuity"] = torch.cat([s["continuity"] for s in font_samples], dim=0)

        if all("alignment" in s for s in font_samples):
            out["alignment"] = torch.cat([s["alignment"] for s in font_samples], dim=0)

        out["glyph_names"] = font_samples[0].get("glyph_names")
        return out

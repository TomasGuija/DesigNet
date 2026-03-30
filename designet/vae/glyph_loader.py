from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import torch

from designet.difflib.tensor import SVGTensor
from designet.svglib.geom import Bbox, Point
from designet.svglib.svg import SVG

RTOL = 0.05
ATOL = 0.01

PAD_VAL = -1


@dataclass
class GlyphLoaderConfig:
    max_num_groups: int
    max_seq_len: int
    normalize: bool = False
    center: bool = False
    compute_continuity: bool = False
    compute_line_alignment: bool = False


def compute_line_alignment_tensor(cmds: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
    """
    For each command, check if it's a LINE aligned with:
      - class 0: horizontal
      - class 1: vertical
      - else: 2 (not aligned or not a line)

    Inputs:
        cmds: (N,) tensor
        args: (N, D) tensor
    Returns:
        alignment_tensor: (N,) tensor with values in {0, 1, 2}
    """

    N = cmds.shape[0]
    alignment = torch.full((N,), -1, dtype=torch.long)

    for i in range(1, N):
        if cmds[i] != 1:
            continue

        x1, y1 = args[i - 1][-2:].tolist()
        x2, y2 = args[i][-2:].tolist()
        dx, dy = x2 - x1, y2 - y1

        if dx == 0 and dy == 0:
            alignment[i] = 2
        elif dy == 0:
            alignment[i] = 0
        elif dx == 0:
            alignment[i] = 1
        else:
            alignment[i] = 2

    return alignment


def compute_continuity_tensor(cmds: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
    """
    Computes continuity classification (C0, C1, C2) at each command endpoint.

    Inputs:
        cmds: Tensor of shape (N,), representing command types (e.g., 0=MOVE, 1=LINE, 2=CUBIC, 3=CLOSE, 4=START)
        args: Tensor of shape (N, 6), each row: [ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y, end_x, end_y]

        We assume there is SOS
    Returns:
        continuity_tensor: Tensor of shape (N,), with:
            0 → C0
            1 → G1
            2 → C1
    """

    def compute_angle(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        v1_norm = v1 / v1.norm().clamp(min=eps)
        v2_norm = v2 / v2.norm().clamp(min=eps)
        dot_product = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
        return torch.acos(dot_product)

    def check_continuity(prev_dir: torch.Tensor, new_dir: torch.Tensor, eps: float = 5e-2) -> int:
        if prev_dir.norm() < 1e-6 or new_dir.norm() < 1e-6:
            return 0

        length_match = torch.isclose(prev_dir.norm(), new_dir.norm(), rtol=RTOL, atol=ATOL)
        angle = compute_angle(prev_dir, new_dir)

        if angle < eps and length_match:
            return 2
        if angle < eps:
            return 1
        return 0

    N = cmds.shape[0]
    continuity = torch.full((N,), -1, dtype=torch.long)

    # Assumes SOS at 0 and first real MOVE at 1
    last_endpoint = args[1][-2:]

    for i in range(2, N - 1):
        if cmds[i] == 3:  # EOS / close depending on your encoding
            break

        current_endpoint = args[i][-2:]

        if cmds[i + 1] == 3:
            next_command = cmds[2]
            next_args = args[2]
        else:
            next_command = cmds[i + 1]
            next_args = args[i + 1]

        if cmds[i] == 1:
            continuity[i] = -1
            if next_command == 2:
                ctrl1 = next_args[:2]
                prev_dir = current_endpoint - last_endpoint
                new_dir = ctrl1 - current_endpoint
                continuity[i] = check_continuity(prev_dir, new_dir)

        elif cmds[i] == 2:
            continuity[i] = 0
            ctrl2 = args[i][2:-2]
            prev_dir = current_endpoint - ctrl2

            if next_command == 1:
                next_endpoint = next_args[-2:]
                new_dir = next_endpoint - current_endpoint
                continuity[i] = check_continuity(prev_dir, new_dir)
            elif next_command == 2:
                ctrl1 = next_args[:2]
                new_dir = ctrl1 - current_endpoint
                continuity[i] = check_continuity(prev_dir, new_dir)

        last_endpoint = current_endpoint

    return continuity


class GlyphLoader:
    """Loads glyph SVG files and converts them into model-ready tensors."""

    def __init__(self, font_dir: str | Path, cfg: GlyphLoaderConfig):
        self.font_dir = Path(font_dir)
        self.cfg = cfg

        if not self.font_dir.exists():
            raise FileNotFoundError(f"Font directory not found: {self.font_dir}")

        svg_paths = sorted(self.font_dir.glob("*.svg"))
        if not svg_paths:
            raise ValueError(f"No SVG files found in {self.font_dir}")

        self.name_to_path = {p.stem: p for p in svg_paths}

    def available_glyphs(self) -> list[str]:
        return sorted(self.name_to_path.keys())

    def preprocess(self, svg: SVG) -> SVG:
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
                center_x = bbox.xy.x + bbox.size.x / 2
                center_y = bbox.xy.y + bbox.size.y / 2
                svg = svg.translate(Point(-center_x, -center_y))

        return svg

    def _load_svg_as_model_input(self, svg_path: Path) -> dict:
        svg = SVG.load_svg(str(svg_path))
        svg = self.preprocess(svg)

        t_sep, fillings = svg.to_tensor(concat_groups=False, PAD_VAL=PAD_VAL), svg.to_fillings()

        if not t_sep:
            raise ValueError(f"No valid tensor groups found in {svg_path}")

        t_sep = list(t_sep)
        fillings = list(fillings)
        num_args = t_sep[0].shape[1]

        # Pad missing groups up to model expectation
        pad_len = max(self.cfg.max_num_groups - len(t_sep), 0)
        t_sep.extend([torch.empty(0, num_args)] * pad_len)
        fillings.extend([0] * pad_len)

        t_sep_tensors = [
            SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(seq_len=self.cfg.max_seq_len + 2)
            for t, f in zip(t_sep, fillings)
        ]

        commands = torch.stack([t.cmds() for t in t_sep_tensors])
        args = torch.stack([t.args(with_start_pos=True) for t in t_sep_tensors])

        model_inputs: Dict[str, Any] = {
            "commands": self._ensure_batch_dim(commands, expected_ndim_without_batch=2),
            "args": self._ensure_batch_dim(args, expected_ndim_without_batch=3),
        }

        if self.cfg.compute_continuity:
            continuity = torch.stack([compute_continuity_tensor(c, a[..., -6:]) for c, a in zip(commands, args)])
            model_inputs["continuity"] = self._ensure_batch_dim(continuity, expected_ndim_without_batch=2)

        if self.cfg.compute_line_alignment:
            line_alignment = torch.stack(
                [compute_line_alignment_tensor(c, a[..., -6:]) for c, a in zip(commands, args)]
            )
            model_inputs["alignment"] = self._ensure_batch_dim(line_alignment, expected_ndim_without_batch=2)

        return model_inputs

    def get(self, glyph_name: str) -> dict:
        if glyph_name not in self.name_to_path:
            available = ", ".join(self.available_glyphs())
            raise KeyError(f"Glyph '{glyph_name}' not found in {self.font_dir}. Available: {available}")

        return self._load_svg_as_model_input(self.name_to_path[glyph_name])

    def __getitem__(self, glyph_name: str) -> dict:
        return self.get(glyph_name)

    @staticmethod
    def _ensure_batch_dim(x: torch.Tensor, expected_ndim_without_batch: int) -> torch.Tensor:
        if x.ndim == expected_ndim_without_batch:
            return x.unsqueeze(0)
        if x.ndim == expected_ndim_without_batch + 1:
            return x
        raise ValueError(
            f"Unexpected ndim={x.ndim}; expected {expected_ndim_without_batch} or " f"{expected_ndim_without_batch + 1}"
        )

    @staticmethod
    def collate_samples_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("samples must contain at least one element")

        first = samples[0]
        if "commands" not in first or "args" not in first:
            raise KeyError("Each sample must contain 'commands' and 'args'")

        batch: Dict[str, Any] = {
            "commands": torch.cat([s["commands"] for s in samples], dim=0),
            "args": torch.cat([s["args"] for s in samples], dim=0),
        }

        optional_keys = ("continuity", "alignment", "letter_code")
        for key in optional_keys:
            if all(key in s for s in samples):
                batch[key] = torch.cat([s[key] for s in samples], dim=0)

        return batch

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

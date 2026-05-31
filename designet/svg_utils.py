from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from designet.difflib.utils import build_svg_from_pred_cmds
from designet.geometry import compute_continuity_tensor, compute_line_alignment_tensor
from designet.svglib.geom import Point
from designet.svglib.svg import SVG
from designet.tensor_utils import (
    PAD_VAL,
    build_svgtensors,
    ensure_batch_dim,
    stack_svgtensors,
)


def to_cp(x: int | str) -> int:
    """Convert a character, decimal string, hex string, or U+XXXX string to a Unicode codepoint."""
    if isinstance(x, int):
        return x

    if isinstance(x, str):
        x = x.strip()

        if len(x) == 1:
            return ord(x)

        x_upper = x.upper().replace("U+", "").replace("0X", "")

        try:
            return int(x_upper, 16)
        except ValueError:
            return int(x)

    raise TypeError(f"Unsupported unicode value: {type(x)}")


def center_svg(svg: SVG) -> SVG:
    """Center an SVG around its bounding-box center."""
    bbox = svg.bbox()
    if bbox is None:
        return svg

    center_x = bbox.xy.x + bbox.size.x / 2
    center_y = bbox.xy.y + bbox.size.y / 2

    return svg.translate(Point(-center_x, -center_y))


def index_svg_paths(font_dir: str | Path, *, require_non_empty: bool = False) -> Dict[str, Path]:
    """Build a glyph-name -> SVG path index using SVG filename stems."""
    font_dir = Path(font_dir)

    if not font_dir.exists():
        raise FileNotFoundError(f"Font directory not found: {font_dir}")

    svg_paths = sorted(font_dir.glob("*.svg"))

    if require_non_empty and not svg_paths:
        raise ValueError(f"No SVG files found in {font_dir}")

    name_to_path: Dict[str, Path] = {}

    for p in svg_paths:
        if p.stem not in name_to_path:
            name_to_path[p.stem] = p

    return name_to_path


def load_svg_as_tensor_sample(
    svg_path: str | Path,
    *,
    max_num_groups: int,
    max_seq_len: int,
    pad_val: int | float = PAD_VAL,
    center: bool = True,
    compute_continuity: bool = False,
    compute_line_alignment: bool = False,
    compute_auxiliary_points: bool = False,
    batch: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load one SVG file and return grouped command/argument tensors."""
    svg_path = Path(svg_path)

    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_path}")

    svg = SVG.load_svg(str(svg_path))

    if center:
        svg = center_svg(svg)

    t_sep = svg.to_tensor(concat_groups=False, PAD_VAL=pad_val)

    if not t_sep:
        raise ValueError(f"No valid tensor groups found in {svg_path}")

    tensors = build_svgtensors(
        t_sep,
        max_num_groups=max_num_groups,
        max_seq_len=max_seq_len,
        pad_val=pad_val,
    )

    out = stack_svgtensors(tensors, with_start_pos=True)

    commands = out["commands"]
    args = out["args"]

    if compute_continuity:
        out["continuity"] = torch.stack([compute_continuity_tensor(c, a[..., -6:]) for c, a in zip(commands, args)])

    if compute_line_alignment:
        out["alignment"] = torch.stack([compute_line_alignment_tensor(c, a[..., -6:]) for c, a in zip(commands, args)])

    if compute_auxiliary_points:
        out["aux_points"] = torch.stack([tensor.sample_auxiliary_points() for tensor in tensors])

    if batch:
        out["commands"] = ensure_batch_dim(out["commands"], expected_ndim_without_batch=2)
        out["args"] = ensure_batch_dim(out["args"], expected_ndim_without_batch=3)

        if "continuity" in out:
            out["continuity"] = ensure_batch_dim(out["continuity"], expected_ndim_without_batch=2)

        if "alignment" in out:
            out["alignment"] = ensure_batch_dim(out["alignment"], expected_ndim_without_batch=2)

        if "aux_points" in out:
            out["aux_points"] = ensure_batch_dim(out["aux_points"], expected_ndim_without_batch=4)

    return out


def _select_single_item_if_batched(
    commands: torch.Tensor,
    args: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align tensors to the unbatched shape expected by build_svg_from_pred_cmds.

    Accepts:
      - commands: (G, S) or (1, G, S)
      - args: (G, S, A) or (1, G, S, A)
    """
    if commands.ndim == 3:
        if commands.shape[0] != 1:
            raise ValueError(
                "Batched commands with batch size > 1 are not supported for single SVG "
                "rendering. Slice one sample before calling svg_from_cmd_args."
            )
        commands = commands[0]
    elif commands.ndim != 2:
        raise ValueError(f"Unexpected commands shape {tuple(commands.shape)}; expected (G,S) or (1,G,S).")

    if args.ndim == 4:
        if args.shape[0] != 1:
            raise ValueError(
                "Batched args with batch size > 1 are not supported for single SVG "
                "rendering. Slice one sample before calling svg_from_cmd_args."
            )
        args = args[0]
    elif args.ndim != 3:
        raise ValueError(f"Unexpected args shape {tuple(args.shape)}; expected (G,S,A) or (1,G,S,A).")

    return commands, args


def svg_from_cmd_args(
    commands: torch.Tensor,
    args: torch.Tensor,
):
    """Build an SVG object from command/argument tensors."""
    commands, args = _select_single_item_if_batched(commands, args)

    svg = build_svg_from_pred_cmds(commands, args, allow_empty=True)

    if svg is None:
        return None

    if svg.empty() or svg.bbox() is None:
        return None

    return svg


def plot_svg(
    svg,
    *,
    ax=None,
    title: Optional[str] = None,
    show_handles: bool = True,
):
    """Plot a single SVG and return ``(fig, ax)``."""
    from matplotlib import pyplot as plt

    owns_axis = ax is None
    if owns_axis:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    svg.draw_matplotlib(ax, show_handles=show_handles)

    if title is not None:
        ax.set_title(title)
    ax.axis("off")

    if owns_axis:
        fig.tight_layout()

    return fig, ax


def plot_svg_comparison(
    left_svg,
    right_svg,
    *,
    left_title: str = "Left",
    right_title: str = "Right",
    show_handles: bool = True,
    figsize: tuple[float, float] = (15, 5),
):
    """Plot two SVGs side by side and return ``(fig, axs)``."""
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    plot_svg(left_svg, ax=axs[0], title=left_title, show_handles=show_handles)
    plot_svg(right_svg, ax=axs[1], title=right_title, show_handles=show_handles)

    fig.tight_layout()
    return fig, axs


def plot_reconstruction(
    sample: Dict,
    reconstruction: Dict,
    *,
    show_handles: bool = True,
    left_title: str = "Original",
    right_title: str = "Reconstruction",
    figsize: tuple[float, float] = (15, 5),
):
    """Convert tensors to SVGs and plot original vs reconstruction."""
    if "commands" not in sample or "args" not in sample:
        raise KeyError("Sample must contain 'commands' and 'args'")

    sample_svg = svg_from_cmd_args(sample["commands"], sample["args"])

    if "commands" not in reconstruction or "args" not in reconstruction:
        raise KeyError("Reconstruction must contain 'commands' and 'args'")

    recon_svg = svg_from_cmd_args(reconstruction["commands"], reconstruction["args"])

    fig, axs = plot_svg_comparison(
        sample_svg,
        recon_svg,
        left_title=left_title,
        right_title=right_title,
        show_handles=show_handles,
        figsize=figsize,
    )

    return {
        "sample_svg": sample_svg,
        "recon_svg": recon_svg,
        "fig": fig,
        "axs": axs,
    }


def plot_word_from_glyph_set(
    glyph_commands: torch.Tensor,
    glyph_args: torch.Tensor,
    *,
    word: str,
    glyph_names: Sequence[str],
    show_handles: bool = True,
    panel_size: tuple[float, float] = (1.8, 2.2),
    title: Optional[str] = None,
):
    """Plot one word from a glyph set in a single row."""
    from matplotlib import pyplot as plt

    if not word:
        raise ValueError("word must be a non-empty string")

    name_to_idx = {str(g): i for i, g in enumerate(glyph_names)}
    indices: List[int] = []
    for ch in word:
        if ch not in name_to_idx:
            raise KeyError(f"Glyph '{ch}' not found in glyph_names")
        indices.append(name_to_idx[ch])

    n = len(indices)
    fig, axs = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]), squeeze=False)

    for j, idx in enumerate(indices):
        svg = svg_from_cmd_args(glyph_commands[idx], glyph_args[idx])
        svg.draw_matplotlib(axs[0][j], show_handles=show_handles)
        axs[0][j].axis("off")

    if title is None:
        title = word
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.subplots_adjust(wspace=max(0.0, float(0.02)))
    return fig, axs


def save_svg_from_cmd_args(
    commands: torch.Tensor,
    args: torch.Tensor,
    output_path: str | Path,
):
    """Convert command/arg tensors into an SVG and save it to disk."""
    svg = svg_from_cmd_args(commands, args)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg.save_svg(output_path)

    return {
        "svg": svg,
        "path": output_path,
    }

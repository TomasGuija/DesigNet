from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd
import torch

from designet.difflib.tensor import SVGTensor

PAD_VAL = -1


def to_device(value: Any, device: str | torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    return value


def ensure_bgs(logits: torch.Tensor, args: torch.Tensor, name: str = "logits") -> torch.Tensor:
    """
    Ensure logits are shaped (B,G,S,C) to match args shaped (B,G,S,D).
    Accepts either (B,G,S,C) or (B,S,G,C).
    """
    if args.ndim != 4:
        raise ValueError(f"args must be 4D (B,G,S,D), got shape {tuple(args.shape)}")

    b, g, s, _ = args.shape
    if logits.ndim != 4:
        raise ValueError(f"{name} must be 4D, got shape {tuple(logits.shape)}")

    if logits.shape[:3] == (b, g, s):
        return logits
    if logits.shape[:3] == (b, s, g):
        return logits.permute(0, 2, 1, 3).contiguous()

    raise RuntimeError(
        f"{name} has shape {tuple(logits.shape)} but expected " f"(B,G,S,C)={(b, g, s)} or (B,S,G,C)={(b, s, g)}"
    )


def build_svgtensors(
    t_sep: Sequence[torch.Tensor],
    *,
    max_num_groups: int,
    max_seq_len: int,
    pad_val: int | float = PAD_VAL,
) -> list[SVGTensor]:
    """Convert separated SVG path tensors into padded SVGTensor objects."""
    t_sep = list(t_sep)

    if not t_sep:
        raise ValueError("t_sep must contain at least one SVG path group")

    if len(t_sep) > max_num_groups:
        raise ValueError(f"SVG has {len(t_sep)} groups, but max_num_groups={max_num_groups}")

    num_args = t_sep[0].shape[1]
    pad_len = max_num_groups - len(t_sep)

    t_sep.extend(
        [
            torch.empty(
                0,
                num_args,
                dtype=t_sep[0].dtype,
                device=t_sep[0].device,
            )
            for _ in range(pad_len)
        ]
    )

    return [SVGTensor.from_data(t, PAD_VAL=pad_val).add_eos().add_sos().pad(seq_len=max_seq_len + 2) for t in t_sep]


def stack_svgtensors(
    tensors: Sequence[SVGTensor],
    *,
    with_start_pos: bool = True,
) -> Dict[str, torch.Tensor]:
    """Stack SVGTensor objects into command and argument tensors."""
    return {
        "commands": torch.stack([t.cmds() for t in tensors]),
        "args": torch.stack([t.args(with_start_pos=with_start_pos) for t in tensors]),
    }


def ensure_batch_dim(x: torch.Tensor, expected_ndim_without_batch: int) -> torch.Tensor:
    """Add a leading batch dimension if it is missing."""
    if x.ndim == expected_ndim_without_batch:
        return x.unsqueeze(0)

    if x.ndim == expected_ndim_without_batch + 1:
        return x

    raise ValueError(
        f"Unexpected ndim={x.ndim}; expected {expected_ndim_without_batch} " f"or {expected_ndim_without_batch + 1}"
    )


def stack_font_glyph_samples(
    glyph_samples: Sequence[Dict[str, torch.Tensor]],
    *,
    batch: bool = False,
    command_key: str = "commands",
    args_key: str = "args",
) -> Dict[str, torch.Tensor]:
    """Stack per-glyph samples into one font-level sample."""
    if not glyph_samples:
        raise ValueError("glyph_samples must contain at least one element")

    commands = torch.stack([g["commands"] for g in glyph_samples], dim=0)
    args = torch.stack([g["args"] for g in glyph_samples], dim=0)

    if batch:
        commands = commands.unsqueeze(0)
        args = args.unsqueeze(0)

    out: Dict[str, torch.Tensor] = {
        command_key: commands,
        args_key: args,
    }

    for key in ("continuity", "alignment"):
        if all(key in g for g in glyph_samples):
            value = torch.stack([g[key] for g in glyph_samples], dim=0)
            if batch:
                value = value.unsqueeze(0)
            out[key] = value

    return out


def collate_stack_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    tensor_keys: Sequence[str] = ("commands", "args"),
    meta_keys: Sequence[str] = (),
) -> Dict[str, Any]:
    """Collate dataset-style samples by stacking tensors along a new batch dimension."""
    if not samples:
        raise ValueError("samples must contain at least one element")

    batch: Dict[str, Any] = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in tensor_keys}

    for key in meta_keys:
        batch[key] = [sample[key] for sample in samples]

    return batch


def collate_cat_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    tensor_keys: Sequence[str] = ("commands", "args"),
    optional_keys: Sequence[str] = ("continuity", "alignment", "letter_code"),
) -> Dict[str, Any]:
    """Collate loader-style samples by concatenating existing batch dimensions."""
    if not samples:
        raise ValueError("samples must contain at least one element")

    batch: Dict[str, Any] = {key: torch.cat([sample[key] for sample in samples], dim=0) for key in tensor_keys}

    for key in optional_keys:
        if all(key in sample for sample in samples):
            batch[key] = torch.cat([sample[key] for sample in samples], dim=0)

    return batch


def check_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Raise if required metadata columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")


def sequence_length_mask(
    df: pd.DataFrame,
    *,
    max_num_groups: int,
    max_seq_len: int,
    max_total_len: int | None = None,
) -> pd.Series:
    """Return mask for rows fitting model sequence limits."""
    check_required_columns(df, ["nb_groups", "max_len_group", "total_len"])

    mask = (df["nb_groups"] <= max_num_groups) & (df["max_len_group"] <= max_seq_len)

    if max_total_len is not None:
        mask = mask & (df["total_len"] <= max_total_len)

    return mask


# Backwards-compatible private aliases for modules that previously imported
# these helpers from designet.vae.tools.
_to_device = to_device
_ensure_bgs = ensure_bgs

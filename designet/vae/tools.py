from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from designet.checkpoint import (
    _HF_VAE_FILE,
    HF_REPO_ID,
    normalize_state_dict_keys,
    resolve_checkpoint_path,
)
from designet.difflib.utils import build_svg_from_pred_cmds
from designet.vae.diff_refinement import (
    SoftAlignmentRefinerBatched,
    SoftContinuityRefinerBatched,
)
from designet.vae.svg_transformer import SVGTransformer


# ---------------------------------------------------------------------
# Checkpoint / config helpers
# ---------------------------------------------------------------------
def load_vae_model(
    checkpoint: Dict[str, Any] | str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[SVGTransformer, Dict[str, Any], Dict[str, Any]]:
    """Load model from a checkpoint. Pass ``None`` to download from HuggingFace Hub."""

    if checkpoint is None or not Path(checkpoint).exists():
        checkpoint = resolve_checkpoint_path(HF_REPO_ID, _HF_VAE_FILE)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["hyper_parameters"]
    cfg["num_groups_proposal"] = cfg["max_num_groups"]
    model = SVGTransformer(cfg)

    raw_state_dict = ckpt["state_dict"]
    state_dict = normalize_state_dict_keys(raw_state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    model.to(device)
    model.eval()

    if strict:
        if missing or unexpected:
            raise RuntimeError(f"Strict load failed.\nMissing keys: {missing}\nUnexpected keys: {unexpected}")
    else:
        if missing:
            print(f"[load warning] Missing keys: {missing}")
        if unexpected:
            print(f"[load warning] Unexpected keys: {unexpected}")

    return model, cfg, ckpt


# ---------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------
def _to_device(value: Any, device: str | torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def _ensure_bgs(logits: torch.Tensor, args: torch.Tensor, name: str = "logits") -> torch.Tensor:
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


# ---------------------------------------------------------------------
# Refinement module
# ---------------------------------------------------------------------
@torch.no_grad()
def refine_output_with_soft_refinement(
    output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply alignment+continuity soft refinement to a model output dict.

    Required output keys:
      - command_logits
      - args_logits
      - cont_logits
      - alignment_logits

    Returns:
      A copy of output with refined args in:
      - args_logits
      - refined_args_logits
    """
    required = ("command_logits", "args_logits", "cont_logits", "alignment_logits")
    missing = [k for k in required if k not in output]
    if missing:
        raise KeyError(f"Output is missing refinement keys: {missing}")

    pred_cmds = output["command_logits"].argmax(dim=-1)
    pred_args = output["args_logits"]

    if pred_args.ndim != 4:
        raise ValueError(
            "Refinement expects continuous args with shape (B,G,S,D). " f"Got shape {tuple(pred_args.shape)}."
        )

    if pred_cmds.ndim == 3 and pred_cmds.shape[:3] == (pred_args.shape[0], pred_args.shape[2], pred_args.shape[1]):
        pred_cmds = pred_cmds.permute(0, 2, 1).contiguous()
    elif pred_cmds.ndim != 3 or pred_cmds.shape[:3] != pred_args.shape[:3]:
        raise RuntimeError(
            f"command predictions shape {tuple(pred_cmds.shape)} is incompatible "
            f"with args shape {tuple(pred_args.shape)}"
        )

    align_logits = _ensure_bgs(output["alignment_logits"], pred_args, name="alignment_logits")
    cont_logits = _ensure_bgs(output["cont_logits"], pred_args, name="cont_logits")

    aligner = SoftAlignmentRefinerBatched(tau=1.0)
    refiner = SoftContinuityRefinerBatched(tau=1.0)

    refined_args = aligner(pred_cmds, pred_args, align_logits)

    cont6 = refiner(pred_cmds, refined_args[..., -6:], cont_logits)
    refined_args[..., -6:-2] = cont6[..., :-2]

    refined_output = dict(output)
    refined_output["args_logits"] = refined_args
    refined_output["refined_args_logits"] = refined_args
    return refined_output


# ---------------------------------------------------------------------
# Inference APIs
# ---------------------------------------------------------------------
@torch.no_grad()
def encode_sample(
    model: SVGTransformer,
    sample: Dict[str, Any],
    device: str | torch.device = "cpu",
):
    sample_dev = _to_device(sample, device)
    model_inputs = {
        "commands": sample_dev["commands"],
        "args": sample_dev["args"],
    }

    return model(**model_inputs, encode_mode=True, return_tgt=False)


@torch.no_grad()
def reconstruct(
    model: SVGTransformer,
    sample: Dict[str, Any],
    device: str | torch.device = "cpu",
    close_paths: bool = True,
):
    """
    Reconstruct from model inputs using greedy decoding.

    Works for both single samples and already-collated batches as long as
    the input dict contains "commands" and "args" in model-expected shapes.
    """
    model_inputs = _to_device(sample, device)

    commands_y, args_y = model.greedy_sample(
        commands=model_inputs["commands"],
        args=model_inputs["args"],
        close_paths=close_paths,
    )

    return {
        "commands": commands_y,
        "args": args_y,
    }


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
    """
    Build an SVG object from command/argument tensors.

    Args:
        commands: Command tensor (G,S) or optionally batched (1,G,S).
        args: Args tensor (G,S,A) or optionally batched (1,G,S,A).
        tighten_viewbox: If True, call SVG.tighten_viewbox() before returning.
        allow_empty: Passed through to build_svg_from_pred_cmds.

    Returns:
        SVG instance.

    Raises:
        ValueError if SVG is empty or invalid.
    """
    commands, args = _select_single_item_if_batched(commands, args)

    svg = build_svg_from_pred_cmds(commands, args, allow_empty=True)

    if svg is None:
        raise ValueError("SVG is empty or invalid")

    svg.tighten_viewbox()

    if svg.empty() or svg.bbox() is None:
        raise ValueError("SVG is empty or invalid")

    return svg


def plot_svg(
    svg,
    *,
    ax=None,
    title: Optional[str] = None,
    show_handles: bool = True,
):
    """
    Plot a single SVG.

    Returns:
        (fig, ax)
    """
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
    """
    Plot two SVGs side by side.

    Returns:
        (fig, axs)
    """
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    plot_svg(
        left_svg,
        ax=axs[0],
        title=left_title,
        show_handles=show_handles,
    )
    plot_svg(
        right_svg,
        ax=axs[1],
        title=right_title,
        show_handles=show_handles,
    )

    fig.tight_layout()
    return fig, axs


def plot_reconstruction(
    sample: Dict[str, Any],
    reconstruction: Dict[str, Any],
    *,
    show_handles: bool = True,
    left_title: str = "Original",
    right_title: str = "Reconstruction",
    figsize: tuple[float, float] = (15, 5),
):
    """
    Convenience helper that converts tensors to SVGs and plots original vs reconstruction.

    Returns:
        dict containing SVGs and matplotlib figure/axes.
    """
    if "commands" not in sample or "args" not in sample:
        raise KeyError("Sample must contain 'commands' and 'args'")

    sample_svg = svg_from_cmd_args(
        sample["commands"],
        sample["args"],
    )

    if "commands" not in reconstruction or "args" not in reconstruction:
        raise KeyError("Reconstruction must contain 'commands' and 'args'")

    recon_svg = svg_from_cmd_args(
        reconstruction["commands"],
        reconstruction["args"],
    )

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
    """
    Convert command/arg tensors into an SVG and save it to disk.

    Returns:
        dict containing the SVG object and resolved output path.
    """
    svg = svg_from_cmd_args(
        commands,
        args,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg.save_svg(output_path)

    return {
        "svg": svg,
        "path": output_path,
    }


@torch.no_grad()
def decode_from_latent(
    model: SVGTransformer,
    z2: torch.Tensor,
    device: str | torch.device = "cpu",
    z1: Optional[torch.Tensor] = None,
    hierarch_logits: Optional[torch.Tensor] = None,
    close_paths: bool = True,
):
    """
    Decode directly from a latent code.
    Useful once you already have z2 (and optionally z1) from encode_sample().
    """
    z2 = _to_device(z2, device)
    z1 = _to_device(z1, device)
    hierarch_logits = _to_device(hierarch_logits, device)

    commands_y, args_y = model.greedy_sample(
        z2=z2,
        z1=z1,
        hierarch_logits=hierarch_logits,
        close_paths=close_paths,
    )

    return {
        "commands": commands_y,
        "args": args_y,
    }


@torch.no_grad()
def interpolate_two_glyphs_linear(
    model: SVGTransformer,
    sample_a: Dict[str, Any],
    sample_b: Dict[str, Any],
    *,
    alphas: Sequence[float],
    device: str | torch.device = "cpu",
    close_paths: bool = True,
) -> Dict[str, Any]:
    """
    Interpolate between two single glyph samples in latent space.

    This is the minimal interpolation API: callers can iterate externally over
    many glyphs/fonts and call this function per pair.
    """
    if not alphas:
        raise ValueError("alphas must contain at least one value")
    alpha_values = [float(a) for a in alphas]
    for a in alpha_values:
        if a < 0.0 or a > 1.0:
            raise ValueError(f"Interpolation alpha must be in [0,1], got {a}")

    encoded_a = encode_sample(model, sample_a, device=device)
    encoded_b = encode_sample(model, sample_b, device=device)

    if isinstance(encoded_a, tuple):
        z_a, z_path_a = encoded_a
    else:
        z_a, z_path_a = encoded_a, None

    if isinstance(encoded_b, tuple):
        z_b, z_path_b = encoded_b
    else:
        z_b, z_path_b = encoded_b, None

    def interp_tensor(a: Optional[torch.Tensor], b: Optional[torch.Tensor], alpha: float):
        if a is None and b is None:
            return None
        if a is None or b is None:
            raise ValueError("Both tensors must be present (or both None) for interpolation")
        if a.shape != b.shape:
            raise ValueError(
                f"Interpolation tensors must have the same shape; got {tuple(a.shape)} vs {tuple(b.shape)}"
            )
        return (1.0 - alpha) * a + alpha * b

    outputs = []
    svgs = []
    for alpha in alpha_values:
        z_i = interp_tensor(z_a, z_b, alpha)
        z_path_i = interp_tensor(z_path_a, z_path_b, alpha)

        pred = decode_from_latent(
            model=model,
            z2=z_i,
            z1=z_path_i,
            device=device,
            close_paths=close_paths,
        )
        outputs.append({"alpha": alpha, **pred})

        svg = svg_from_cmd_args(
            pred["commands"],
            pred["args"],
        )
        svgs.append(svg)

    return {
        "alphas": alpha_values,
        "outputs": outputs,
        "svgs": svgs,
        "sample_a": sample_a,
        "sample_b": sample_b,
    }

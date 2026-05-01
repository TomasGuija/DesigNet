from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from designet.checkpoint import (
    _HF_VAE_FILE,
    HF_REPO_ID,
    normalize_state_dict_keys,
    resolve_checkpoint_path,
)
from designet.svg_utils import (
    plot_reconstruction,
    plot_svg,
    plot_svg_comparison,
    plot_word_from_glyph_set,
    save_svg_from_cmd_args,
    svg_from_cmd_args,
)
from designet.tensor_utils import _ensure_bgs, _to_device
from designet.vae.diff_refinement import (
    SoftAlignmentRefinerBatched,
    SoftContinuityRefinerBatched,
)
from designet.vae.svg_transformer import SVGTransformer


def load_vae_model(
    checkpoint: Dict[str, Any] | str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> tuple[SVGTransformer, Dict[str, Any], Dict[str, Any]]:
    """Load a VAE checkpoint. Pass ``None`` to use the default Hugging Face checkpoint."""
    if isinstance(checkpoint, dict):
        ckpt = checkpoint
    else:
        if checkpoint is None or not Path(checkpoint).exists():
            checkpoint = resolve_checkpoint_path(HF_REPO_ID, _HF_VAE_FILE)

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["hyper_parameters"]
    cfg["num_groups_proposal"] = cfg["max_num_groups"]
    model = SVGTransformer(cfg)

    state_dict = normalize_state_dict_keys(ckpt["state_dict"])
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


@torch.no_grad()
def refine_output_with_soft_refinement(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply alignment and continuity soft refinement to a VAE output dictionary.
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


@torch.no_grad()
def encode_sample(
    model: SVGTransformer,
    sample: Dict[str, Any],
    device: str | torch.device = "cpu",
):
    sample_dev = _to_device(sample, device)
    return model(
        commands=sample_dev["commands"],
        args=sample_dev["args"],
        encode_mode=True,
        return_tgt=False,
    )


@torch.no_grad()
def reconstruct(
    model: SVGTransformer,
    sample: Dict[str, Any],
    device: str | torch.device = "cpu",
    close_paths: bool = True,
):
    """Reconstruct model inputs using greedy decoding."""
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


@torch.no_grad()
def decode_from_latent(
    model: SVGTransformer,
    z2: torch.Tensor,
    device: str | torch.device = "cpu",
    z1: Optional[torch.Tensor] = None,
    hierarch_logits: Optional[torch.Tensor] = None,
    close_paths: bool = True,
):
    """Decode directly from latent codes returned by ``encode_sample``."""
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
    """Linearly interpolate between two single-glyph samples in latent space."""
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
            raise ValueError("Both tensors must be present, or both None, for interpolation")
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
        svgs.append(svg_from_cmd_args(pred["commands"], pred["args"]))

    return {
        "alphas": alpha_values,
        "outputs": outputs,
        "svgs": svgs,
        "sample_a": sample_a,
        "sample_b": sample_b,
    }


__all__ = [
    "load_vae_model",
    "refine_output_with_soft_refinement",
    "encode_sample",
    "reconstruct",
    "decode_from_latent",
    "interpolate_two_glyphs_linear",
    "svg_from_cmd_args",
    "save_svg_from_cmd_args",
    "plot_svg",
    "plot_svg_comparison",
    "plot_reconstruction",
    "plot_word_from_glyph_set",
]

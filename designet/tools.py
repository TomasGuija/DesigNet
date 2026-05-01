from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from designet.checkpoint import (
    _HF_DESIGNET_FILE,
    HF_REPO_ID,
    normalize_state_dict_keys,
    resolve_checkpoint_path,
)
from designet.geometry import compute_continuity_tensor, compute_line_alignment_tensor
from designet.model import FontConditionalSVGTransformer
from designet.svg_utils import (
    center_svg,
    index_svg_paths,
    load_svg_as_tensor_sample,
    svg_from_cmd_args,
    to_cp,
)
from designet.tensor_utils import (
    PAD_VAL,
    _ensure_bgs,
    _to_device,
    build_svgtensors,
    check_required_columns,
    collate_cat_samples,
    collate_stack_samples,
    ensure_batch_dim,
    sequence_length_mask,
    stack_font_glyph_samples,
    stack_svgtensors,
)
from designet.vae.diff_refinement import (
    SoftAlignmentRefinerBatched,
    SoftContinuityRefinerBatched,
)


def fix_positional_encoding_shapes(state_dict, model):
    """Adapt older positional encoding tensors to the current checkpoint layout."""
    fixed = dict(state_dict)

    for key, value in list(fixed.items()):
        if key not in model.state_dict():
            continue

        target = model.state_dict()[key]
        if value.shape == target.shape:
            continue

        if value.ndim == 3 and target.ndim == 2 and value.shape[1] == 1:
            squeezed = value.squeeze(1)
            if squeezed.shape == target.shape:
                fixed[key] = squeezed
                print(f"[shape fix] squeezed {key}: {tuple(value.shape)} -> {tuple(squeezed.shape)}")

    return fixed


def load_designet_model(
    checkpoint: Dict[str, Any] | str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[FontConditionalSVGTransformer, Dict[str, Any], Dict[str, Any]]:
    """Load a DesigNet checkpoint. Pass ``None`` to use the default Hugging Face checkpoint."""
    if isinstance(checkpoint, dict):
        ckpt = checkpoint
    else:
        if checkpoint is None:
            checkpoint = resolve_checkpoint_path(HF_REPO_ID, _HF_DESIGNET_FILE)

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["hyper_parameters"]
    cfg["n_commands"] = 5
    cfg["num_encoding_glyphs"] = len(cfg["encoding_glyphs"])
    cfg["num_decoding_glyphs"] = len(cfg["decoding_glyphs"])
    model = FontConditionalSVGTransformer(cfg)

    state_dict = normalize_state_dict_keys(ckpt["state_dict"])
    state_dict = fix_positional_encoding_shapes(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    model.to(device)
    model.eval()

    if strict and (missing or unexpected):
        raise RuntimeError(f"Strict load failed. Missing keys: {missing}. Unexpected keys: {unexpected}.")

    if not strict:
        if missing:
            print(f"[load warning] Missing keys: {missing}")
        if unexpected:
            print(f"[load warning] Unexpected keys: {unexpected}")

    return model, cfg, ckpt


def _build_conditioning_from_inputs(
    model: FontConditionalSVGTransformer,
    *,
    input_commands: torch.Tensor,
    input_args: torch.Tensor,
    sel_self: torch.Tensor,
    sel_cross: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build decoder conditioning tensors from model inputs."""
    bsz, _, num_groups, _ = input_commands.shape

    encoding_commands = input_commands[:, : model.num_encoding_letters, :, :].flatten(0, 1)
    encoding_args = input_args[:, : model.num_encoding_letters, :, :, :].flatten(0, 1)

    abs_self = sel_self
    abs_cross = sel_cross + model.num_encoding_letters
    abs_all = torch.cat([abs_self, abs_cross], dim=-1).to(input_commands.device)

    glyph_embedding = model.glyph_emb(abs_all)

    z, z1 = model.base_model.encoder(encoding_commands, encoding_args)
    z1 = z1.squeeze(0).reshape(bsz, model.num_encoding_letters, num_groups, -1)

    cls_token_group = model.cls_token_group[None].expand(bsz, 1, num_groups, -1)
    z1_with_cls = torch.cat([cls_token_group, z1], dim=-3)
    z1_with_cls = z1_with_cls.transpose(1, 2).flatten(0, 1)
    z1_agg = model.aggregator_transformer_group(z1_with_cls)
    z1_agg = z1_agg[:, 0]

    z1_agg = z1_agg.reshape(bsz, num_groups, -1)
    z1, _, _ = model.base_model.vae1(z1_agg)

    z1 = z1.unsqueeze(-3)
    z1 = z1.expand(-1, abs_all.size(-1), -1, -1)

    glyph_embedding_group = glyph_embedding.unsqueeze(-2)
    glyph_embedding_group = glyph_embedding_group.expand_as(z1)
    fused = torch.cat([z1, glyph_embedding_group], dim=-1)
    conditioned_group_embeddings = model.condition_letter_proj_group(fused).flatten(0, 1)

    z = z.view(bsz, model.num_encoding_letters, -1)
    cls_token = model.cls_token.expand(bsz, -1, -1)
    z_with_cls = torch.cat([cls_token, z], dim=1)
    z_agg = model.aggregator_transformer(z_with_cls)
    z = z_agg[:, 0]
    z, _, _ = model.base_model.vae2(z)

    z_exp = z.unsqueeze(1).expand(-1, abs_all.size(1), -1)
    concat_embeddings = torch.cat([z_exp, glyph_embedding], dim=-1)
    conditioned_embeddings = model.condition_letter_proj(concat_embeddings).flatten(0, 1).unsqueeze(1)

    return conditioned_embeddings, conditioned_group_embeddings


def _decode_from_conditioning(
    model: FontConditionalSVGTransformer,
    *,
    conditioned_embeddings: torch.Tensor,
    conditioned_group_embeddings: Optional[torch.Tensor],
    sel_self: torch.Tensor,
    sel_cross: torch.Tensor,
    close_paths: bool,
) -> Dict[str, Any]:
    """Decode from precomputed conditioning tensors into full-font tensors."""
    bsz = sel_self.shape[0]
    k_self = sel_self.shape[1]
    k_cross = sel_cross.shape[1]
    n_letters = k_self + k_cross

    out_logits = model.base_model.decoder(conditioned_embeddings, z_path=conditioned_group_embeddings)
    out_logits = {k: v.unflatten(0, (bsz, n_letters)) for k, v in out_logits.items()}

    def split_logits(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_self = x[:, :k_self].reshape(bsz * k_self, *x.shape[2:])
        x_cross = x[:, k_self:].reshape(bsz * k_cross, *x.shape[2:])
        return x_self, x_cross

    pred = {
        "self_command_logits": split_logits(out_logits["commands"])[0],
        "cross_command_logits": split_logits(out_logits["commands"])[1],
        "self_args_logits": split_logits(out_logits["args"])[0],
        "cross_args_logits": split_logits(out_logits["args"])[1],
        "self_visibility_logits": split_logits(out_logits["visibility"])[0],
        "cross_visibility_logits": split_logits(out_logits["visibility"])[1],
    }

    pred["self_cont_logits"], pred["cross_cont_logits"] = split_logits(out_logits["continuity"])
    pred["self_alignment_logits"], pred["cross_alignment_logits"] = split_logits(out_logits["alignment"])

    cmd_cross, arg_cross, cmd_self, arg_self = model.greedy_sample(
        pred=pred,
        close_paths=close_paths,
        sel_self=sel_self,
        sel_cross=sel_cross,
    )

    cmd_self = cmd_self.unflatten(0, (bsz, k_self))
    arg_self = arg_self.unflatten(0, (bsz, k_self))
    cmd_cross = cmd_cross.unflatten(0, (bsz, k_cross))
    arg_cross = arg_cross.unflatten(0, (bsz, k_cross))

    n_total = model.num_encoding_letters + model.num_decoding_letters
    commands = cmd_self.new_full((bsz, n_total, cmd_self.shape[2], cmd_self.shape[3]), 3)
    args = arg_self.new_full((bsz, n_total, arg_self.shape[2], arg_self.shape[3], arg_self.shape[4]), -1)

    for b in range(bsz):
        commands[b, sel_self[b]] = cmd_self[b]
        args[b, sel_self[b]] = arg_self[b]
        cross_pos = sel_cross[b] + model.num_encoding_letters
        commands[b, cross_pos] = cmd_cross[b]
        args[b, cross_pos] = arg_cross[b]

    return {
        "commands": commands,
        "args": args,
        "commands_self": cmd_self,
        "args_self": arg_self,
        "commands_cross": cmd_cross,
        "args_cross": arg_cross,
        "sel_self": sel_self,
        "sel_cross": sel_cross,
    }


@torch.no_grad()
def reconstruct_font(
    model: FontConditionalSVGTransformer,
    font_sample: Dict[str, Any],
    device: str | torch.device = "cpu",
    close_paths: bool = True,
) -> Dict[str, Any]:
    """Reconstruct full fonts from batched model inputs."""
    sample_dev = _to_device(font_sample, device)

    input_commands = sample_dev["input_commands"]
    input_args = sample_dev["input_args"]
    bsz = input_commands.shape[0]

    sel_self_eff = torch.arange(model.num_encoding_letters, device=input_commands.device).unsqueeze(0).expand(bsz, -1)
    sel_cross_eff = torch.arange(model.num_decoding_letters, device=input_commands.device).unsqueeze(0).expand(bsz, -1)

    pred = model(
        input_commands=input_commands,
        input_args=input_args,
        return_tgt=False,
        sel_self=sel_self_eff,
        sel_cross=sel_cross_eff,
    )

    cmd_cross, arg_cross, cmd_self, arg_self = model.greedy_sample(
        pred=pred,
        close_paths=close_paths,
        sel_self=sel_self_eff,
        sel_cross=sel_cross_eff,
    )

    k_self = sel_self_eff.shape[1]
    k_cross = sel_cross_eff.shape[1]

    cmd_self = cmd_self.unflatten(0, (bsz, k_self))
    arg_self = arg_self.unflatten(0, (bsz, k_self))
    cmd_cross = cmd_cross.unflatten(0, (bsz, k_cross))
    arg_cross = arg_cross.unflatten(0, (bsz, k_cross))

    n_total = model.num_encoding_letters + model.num_decoding_letters
    commands = cmd_self.new_full((bsz, n_total, cmd_self.shape[2], cmd_self.shape[3]), 3)
    args = arg_self.new_full((bsz, n_total, arg_self.shape[2], arg_self.shape[3], arg_self.shape[4]), -1)

    for b in range(bsz):
        commands[b, sel_self_eff[b]] = cmd_self[b]
        args[b, sel_self_eff[b]] = arg_self[b]

        cross_pos = sel_cross_eff[b] + model.num_encoding_letters
        commands[b, cross_pos] = cmd_cross[b]
        args[b, cross_pos] = arg_cross[b]

    return {
        "commands": commands,
        "args": args,
        "commands_self": cmd_self,
        "args_self": arg_self,
        "commands_cross": cmd_cross,
        "args_cross": arg_cross,
        "sel_self": sel_self_eff,
        "sel_cross": sel_cross_eff,
        "pred": pred,
    }


@torch.no_grad()
def interpolate_two_fonts_linear(
    model: FontConditionalSVGTransformer,
    font_sample_a: Dict[str, Any],
    font_sample_b: Dict[str, Any],
    *,
    alphas: Sequence[float],
    device: str | torch.device = "cpu",
    close_paths: bool = True,
) -> Dict[str, Any]:
    """Linearly interpolate between two fonts in latent conditioning space."""
    if not alphas:
        raise ValueError("alphas must contain at least one value")

    alpha_values = [float(a) for a in alphas]
    for a in alpha_values:
        if a < 0.0 or a > 1.0:
            raise ValueError(f"Interpolation alpha must be in [0,1], got {a}")

    a_dev = _to_device(font_sample_a, device)
    b_dev = _to_device(font_sample_b, device)

    input_commands_a = a_dev["input_commands"]
    input_args_a = a_dev["input_args"]
    input_commands_b = b_dev["input_commands"]
    input_args_b = b_dev["input_args"]

    if input_commands_a.shape != input_commands_b.shape:
        raise ValueError(
            "font_sample_a and font_sample_b must have the same input_commands shape; "
            f"got {tuple(input_commands_a.shape)} vs {tuple(input_commands_b.shape)}"
        )
    if input_args_a.shape != input_args_b.shape:
        raise ValueError(
            "font_sample_a and font_sample_b must have the same input_args shape; "
            f"got {tuple(input_args_a.shape)} vs {tuple(input_args_b.shape)}"
        )

    bsz = input_commands_a.shape[0]
    if bsz != 1:
        raise ValueError(f"Expected batch size 1 per font sample, got {bsz}")

    sel_self_eff = torch.arange(model.num_encoding_letters, device=input_commands_a.device).unsqueeze(0).expand(bsz, -1)
    sel_cross_eff = (
        torch.arange(model.num_decoding_letters, device=input_commands_a.device).unsqueeze(0).expand(bsz, -1)
    )

    cond_a, cond_group_a = _build_conditioning_from_inputs(
        model,
        input_commands=input_commands_a,
        input_args=input_args_a,
        sel_self=sel_self_eff,
        sel_cross=sel_cross_eff,
    )
    cond_b, cond_group_b = _build_conditioning_from_inputs(
        model,
        input_commands=input_commands_b,
        input_args=input_args_b,
        sel_self=sel_self_eff,
        sel_cross=sel_cross_eff,
    )

    if cond_a.shape != cond_b.shape:
        raise RuntimeError(
            "Inconsistent conditioning shapes for interpolation: " f"{tuple(cond_a.shape)} vs {tuple(cond_b.shape)}"
        )
    if (cond_group_a is None) != (cond_group_b is None):
        raise RuntimeError("Inconsistent hierarchical conditioning presence across interpolation endpoints")

    interpolations: List[Dict[str, Any]] = []
    for a in alpha_values:
        cond = (1.0 - a) * cond_a + a * cond_b
        cond_group = None if cond_group_a is None else (1.0 - a) * cond_group_a + a * cond_group_b

        recon = _decode_from_conditioning(
            model,
            conditioned_embeddings=cond,
            conditioned_group_embeddings=cond_group,
            sel_self=sel_self_eff,
            sel_cross=sel_cross_eff,
            close_paths=close_paths,
        )
        recon["alpha"] = a
        recon["num_encoding_letters"] = model.num_encoding_letters
        recon["num_decoding_letters"] = model.num_decoding_letters
        interpolations.append(recon)

    return {
        "alphas": alpha_values,
        "interpolations": interpolations,
        "font_sample_a": font_sample_a,
        "font_sample_b": font_sample_b,
        "glyph_names": font_sample_a.get("glyph_names"),
    }


def save_reconstructed_font_svgs(
    reconstruction: Dict[str, Any],
    output_dir: str | Path,
    *,
    glyph_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Save reconstructed font SVGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmds_all = reconstruction["commands"]
    args_all = reconstruction["args"]
    if cmds_all.ndim == 3:
        cmds_all = cmds_all.unsqueeze(0)
    elif cmds_all.ndim != 4:
        raise ValueError(
            "Expected reconstruction['commands'] to be 4D (B,N,G,S) or 3D (N,G,S), " f"got {tuple(cmds_all.shape)}"
        )

    if args_all.ndim == 4:
        args_all = args_all.unsqueeze(0)
    elif args_all.ndim != 5:
        raise ValueError(
            "Expected reconstruction['args'] to be 5D (B,N,G,S,D) or 4D (N,G,S,D), " f"got {tuple(args_all.shape)}"
        )

    if cmds_all.shape[0] != args_all.shape[0] or cmds_all.shape[1:4] != args_all.shape[1:4]:
        raise ValueError(
            "Inconsistent shapes between reconstruction['commands'] and reconstruction['args']: "
            f"{tuple(cmds_all.shape)} vs {tuple(args_all.shape)}"
        )

    batch_size, n, _, _ = cmds_all.shape
    names = list(glyph_names) if glyph_names is not None else [f"glyph_{i:03d}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"Expected {n} glyph names, got {len(names)}")

    def save_one_batch(*, b: int, dst: Path) -> List[Path]:
        dst.mkdir(parents=True, exist_ok=True)
        cmds = cmds_all[b]
        args = args_all[b]
        paths: List[Path] = []
        for i in range(n):
            svg = svg_from_cmd_args(cmds[i], args[i])
            out_path = dst / f"{names[i]}.svg"
            svg.save_svg(out_path)
            paths.append(out_path)
        return paths

    if batch_size == 1:
        return save_one_batch(b=0, dst=output_dir)

    all_saved: List[Path] = []
    for b in range(batch_size):
        all_saved.extend(save_one_batch(b=b, dst=output_dir / f"batch_{b:03d}"))
    return all_saved


@torch.no_grad()
def refine_output_with_soft_refinement(
    output: Dict[str, Any],
    *,
    align_min_conf: float = 0.0,
    cont_min_conf: float = 0.0,
) -> Dict[str, Any]:
    """Apply alignment and continuity soft refinement directly to DesigNet output logits."""
    required = ("self_command_logits", "cross_command_logits", "self_args_logits", "cross_args_logits")
    missing = [k for k in required if k not in output]
    if missing:
        raise KeyError(f"Output is missing required keys: {missing}")

    align_noop_class = 2
    cont_noop_class = 0

    cmd_self = output["self_command_logits"].argmax(dim=-1)
    cmd_cross = output["cross_command_logits"].argmax(dim=-1)
    args_self = output["self_args_logits"]
    args_cross = output["cross_args_logits"]

    if args_self.ndim != 4 or args_cross.ndim != 4:
        raise ValueError(
            "Refinement expects args logits with shape (B,G,S,D). "
            f"Got self={tuple(args_self.shape)}, cross={tuple(args_cross.shape)}"
        )

    if cmd_self.ndim != 3 or cmd_self.shape[:3] != args_self.shape[:3]:
        raise RuntimeError(
            f"self command predictions shape {tuple(cmd_self.shape)} is incompatible "
            f"with self args shape {tuple(args_self.shape)}"
        )
    if cmd_cross.ndim != 3 or cmd_cross.shape[:3] != args_cross.shape[:3]:
        raise RuntimeError(
            f"cross command predictions shape {tuple(cmd_cross.shape)} is incompatible "
            f"with cross args shape {tuple(args_cross.shape)}"
        )

    aligner = SoftAlignmentRefinerBatched(tau=1.0)
    refiner = SoftContinuityRefinerBatched(tau=1.0)

    refined_self = args_self.clone()
    refined_cross = args_cross.clone()

    has_align = "self_alignment_logits" in output and "cross_alignment_logits" in output
    if has_align:
        align_self = _ensure_bgs(output["self_alignment_logits"], args_self, name="self_alignment_logits")
        align_cross = _ensure_bgs(output["cross_alignment_logits"], args_cross, name="cross_alignment_logits")

        if align_min_conf > 0.0:
            for name, x in (("self_alignment_logits", align_self), ("cross_alignment_logits", align_cross)):
                if x.shape[-1] <= align_noop_class:
                    raise ValueError(
                        f"{name} has {x.shape[-1]} classes, align_noop_class={align_noop_class} is out of range"
                    )

            self_probs = align_self.softmax(dim=-1)
            self_conf, self_pred = self_probs.max(dim=-1)
            self_ids = torch.where(
                self_conf >= float(align_min_conf),
                self_pred,
                torch.full_like(self_pred, int(align_noop_class)),
            )
            align_self = torch.nn.functional.one_hot(self_ids, num_classes=align_self.shape[-1]).float()

            cross_probs = align_cross.softmax(dim=-1)
            cross_conf, cross_pred = cross_probs.max(dim=-1)
            cross_ids = torch.where(
                cross_conf >= float(align_min_conf),
                cross_pred,
                torch.full_like(cross_pred, int(align_noop_class)),
            )
            align_cross = torch.nn.functional.one_hot(cross_ids, num_classes=align_cross.shape[-1]).float()

        refined_self = aligner(cmd_self, refined_self, align_self)
        refined_cross = aligner(cmd_cross, refined_cross, align_cross)

    has_cont = "self_cont_logits" in output and "cross_cont_logits" in output
    if has_cont:
        cont_self = _ensure_bgs(output["self_cont_logits"], args_self, name="self_cont_logits")
        cont_cross = _ensure_bgs(output["cross_cont_logits"], args_cross, name="cross_cont_logits")

        if cont_min_conf > 0.0:
            for name, x in (("self_cont_logits", cont_self), ("cross_cont_logits", cont_cross)):
                if x.shape[-1] <= cont_noop_class:
                    raise ValueError(
                        f"{name} has {x.shape[-1]} classes, cont_noop_class={cont_noop_class} is out of range"
                    )

            self_probs = cont_self.softmax(dim=-1)
            self_conf, self_pred = self_probs.max(dim=-1)
            self_ids = torch.where(
                self_conf >= float(cont_min_conf),
                self_pred,
                torch.full_like(self_pred, int(cont_noop_class)),
            )
            cont_self = torch.nn.functional.one_hot(self_ids, num_classes=cont_self.shape[-1]).float()

            cross_probs = cont_cross.softmax(dim=-1)
            cross_conf, cross_pred = cross_probs.max(dim=-1)
            cross_ids = torch.where(
                cross_conf >= float(cont_min_conf),
                cross_pred,
                torch.full_like(cross_pred, int(cont_noop_class)),
            )
            cont_cross = torch.nn.functional.one_hot(cross_ids, num_classes=cont_cross.shape[-1]).float()

        cont6_self = refiner(cmd_self, refined_self[..., -6:], cont_self)
        cont6_cross = refiner(cmd_cross, refined_cross[..., -6:], cont_cross)
        refined_self[..., -6:-2] = cont6_self[..., :-2]
        refined_cross[..., -6:-2] = cont6_cross[..., :-2]

    refined_output = dict(output)
    refined_output["self_args_logits"] = refined_self
    refined_output["cross_args_logits"] = refined_cross
    refined_output["self_refined_args_logits"] = refined_self
    refined_output["cross_refined_args_logits"] = refined_cross
    return refined_output


__all__ = [
    "load_designet_model",
    "reconstruct_font",
    "interpolate_two_fonts_linear",
    "save_reconstructed_font_svgs",
    "refine_output_with_soft_refinement",
    "svg_from_cmd_args",
    "load_svg_as_tensor_sample",
    "index_svg_paths",
    "center_svg",
    "to_cp",
    "PAD_VAL",
    "build_svgtensors",
    "stack_svgtensors",
    "ensure_batch_dim",
    "stack_font_glyph_samples",
    "collate_stack_samples",
    "collate_cat_samples",
    "check_required_columns",
    "sequence_length_mask",
    "compute_continuity_tensor",
    "compute_line_alignment_tensor",
]

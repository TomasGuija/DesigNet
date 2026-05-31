"""Shared loss helper functions for SVG training objectives.

This module contains small reusable geometric terms used by both the standalone VAE
loss and the font-conditional DesigNet loss.
"""

from __future__ import annotations

import torch

from designet.difflib.tensor import SVGTensor

ARG_WIDTH_8 = 8


def compute_endpoint_consistency_loss(
    commands: torch.Tensor,
    args: torch.Tensor,
) -> torch.Tensor:
    """Encourage explicit start/end points in the 8-argument representation to agree."""
    batch_size, num_groups, seq_len, arg_width = args.shape
    if arg_width != ARG_WIDTH_8:
        raise ValueError("Endpoint consistency loss is only valid for 8-argument SVG tensors.")

    start_pos = args[..., :2]
    end_pos = args[..., -2:]
    eos_id = SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
    move_id = SVGTensor.COMMANDS_SIMPLIFIED.index("m")

    valid_next = commands[..., 1:] != eos_id
    dist_next = (end_pos[..., :-1, :] - start_pos[..., 1:, :]).pow(2).sum(dim=-1)
    loss = dist_next[valid_next].sum()
    count = valid_next.sum()

    move_mask = commands[..., 0] == move_id
    valid_nonmove = (commands != eos_id) & (commands != move_id)
    indices = (
        torch.arange(seq_len, device=commands.device)
        .view(1, 1, seq_len)
        .expand(
            batch_size,
            num_groups,
            seq_len,
        )
    )
    last_valid_idx = indices.masked_fill(~valid_nonmove, -1).max(dim=-1).values
    gather_idx = last_valid_idx.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    last_end = end_pos.gather(dim=2, index=gather_idx).squeeze(2)
    move_end = end_pos[:, :, 0, :]

    move_dist = (last_end - move_end).pow(2).sum(dim=-1)
    loss = loss + move_dist[move_mask].sum()
    count = count + move_mask.sum()

    return loss / count.clamp(min=1)


def sample_predicted_auxiliary_points(
    pred_args: torch.Tensor,
    target_commands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample auxiliary points from predicted arguments using ground-truth command types.

    The command targets decide whether each position is sampled as a line or as
    a cubic curve.
    """
    _, _, _, arg_width = pred_args.shape
    if arg_width != ARG_WIDTH_8:
        raise ValueError("Auxiliary point loss expects 8-argument SVG tensors.")

    start = pred_args[..., :2]
    ctrl1 = pred_args[..., 2:4]
    ctrl2 = pred_args[..., 4:6]
    end = pred_args[..., -2:]

    t_vals = pred_args.new_tensor([0.25, 0.5, 0.75]).view(1, 1, 1, 3, 1)
    one_minus_t = 1.0 - t_vals

    line_points = start.unsqueeze(-2) + t_vals * (end - start).unsqueeze(-2)
    curve_points = (
        one_minus_t.pow(3) * start.unsqueeze(-2)
        + 3.0 * one_minus_t.pow(2) * t_vals * ctrl1.unsqueeze(-2)
        + 3.0 * one_minus_t * t_vals.pow(2) * ctrl2.unsqueeze(-2)
        + t_vals.pow(3) * end.unsqueeze(-2)
    )

    line_id = SVGTensor.COMMANDS_SIMPLIFIED.index("l")
    cubic_id = SVGTensor.COMMANDS_SIMPLIFIED.index("c")
    line_mask = target_commands == line_id
    cubic_mask = target_commands == cubic_id
    valid_mask = line_mask | cubic_mask

    pred_points = torch.full_like(line_points, -1.0)
    pred_points = torch.where(line_mask.unsqueeze(-1).unsqueeze(-1), line_points, pred_points)
    pred_points = torch.where(cubic_mask.unsqueeze(-1).unsqueeze(-1), curve_points, pred_points)
    return pred_points, valid_mask

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from designet.difflib.tensor import SVGTensor
from designet.loss_helpers import (
    compute_endpoint_consistency_loss,
    sample_predicted_auxiliary_points,
)
from designet.vae.utils import _get_padding_mask, _get_visibility_mask


class DesigNetLoss(nn.Module):
    """Supervised self/cross reconstruction loss for the font generative model."""

    N_ARGS = 8

    def __init__(self) -> None:
        super().__init__()
        self.n_commands = len(SVGTensor.COMMANDS_SIMPLIFIED)
        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK_8_ARGS)

    def forward(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        weights: Mapping[str, float],
    ) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        loss = output["self_command_logits"].new_zeros(())
        cross_multiplier = weights["cross_reconstruction_multiplier"]
        batch_size = targets["commands"].size(0)
        num_self = output["self_command_logits"].size(0) // batch_size
        num_cross = output["cross_command_logits"].size(0) // batch_size

        self_targets = self._branch_targets(
            targets,
            start=0,
            count=num_self,
            seq_len=output["self_command_logits"].size(-2),
        )
        cross_targets = self._branch_targets(
            targets,
            start=num_self,
            count=num_cross,
            seq_len=output["cross_command_logits"].size(-2),
        )

        self_loss, self_metrics = self._reconstruction_loss(
            output,
            self_targets,
            weights,
            prefix="self",
            multiplier=1.0,
        )
        cross_loss, cross_metrics = self._reconstruction_loss(
            output,
            cross_targets,
            weights,
            prefix="cross",
            multiplier=cross_multiplier,
        )
        loss = loss + self_loss + cross_loss
        metrics.update(self_metrics)
        metrics.update(cross_metrics)

        kl_weight = weights["loss_kl_weight"]
        tolerance = weights["kl_tolerance"]
        kl_path = kl_weight * self._single_kl(output["mu1"], output["logsigma1"], tolerance)
        kl_glyph = kl_weight * self._single_kl(output["mu2"], output["logsigma2"], tolerance)
        loss = loss + kl_path + kl_glyph
        metrics["loss/kl_path"] = kl_path
        metrics["loss/kl_glyph"] = kl_glyph

        metrics["loss/total"] = loss
        return metrics

    def _branch_targets(
        self,
        targets: Mapping[str, torch.Tensor],
        *,
        start: int,
        count: int,
        seq_len: int,
    ) -> Dict[str, torch.Tensor]:
        return {
            "commands": self._flatten_branch_target(targets["commands"], start, count, seq_len),
            "args": self._flatten_branch_target(targets["args"], start, count, seq_len),
            "continuity": self._flatten_branch_target(targets["continuity"], start, count, seq_len),
            "alignment": self._flatten_branch_target(targets["alignment"], start, count, seq_len),
            "aux_points": self._flatten_branch_target(targets["aux_points"], start, count, seq_len),
        }

    @staticmethod
    def _flatten_branch_target(
        target: torch.Tensor,
        start: int,
        count: int,
        seq_len: int,
    ) -> torch.Tensor:
        target = target[:, start : start + count, :, 1:]
        target = target[:, :, :, :seq_len, ...]
        batch_size, n_glyphs, n_groups, seq_len = target.shape[:4]
        return target.reshape(batch_size * n_glyphs, n_groups, seq_len, *target.shape[4:])

    def _reconstruction_loss(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        weights: Mapping[str, float],
        *,
        prefix: str,
        multiplier: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target_commands = targets["commands"].long()
        target_args = targets["args"]
        target_aux_points = targets["aux_points"]
        command_logits = output[f"{prefix}_command_logits"]
        args_logits = output[f"{prefix}_args_logits"]
        visibility_logits = output[f"{prefix}_visibility_logits"]
        loss_cmd_weight = weights["loss_cmd_weight"]
        loss_args_weight = weights["loss_args_weight"]
        loss_visibility_weight = weights["loss_visibility_weight"]
        loss_continuity_weight = weights["loss_continuity_weight"]
        loss_alignment_weight = weights["loss_alignment_weight"]
        loss_consistency_weight = weights["loss_consistency_weight"]
        loss_aux_weight = weights["loss_aux_weight"]

        visibility_mask = _get_visibility_mask(target_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(target_commands, seq_dim=-1, extended=True)
        padding_mask = padding_mask * visibility_mask.unsqueeze(-1)
        valid_tokens = padding_mask.bool()

        loss = command_logits.new_zeros(())
        metrics: Dict[str, torch.Tensor] = {}

        loss_visibility = F.cross_entropy(
            visibility_logits.reshape(-1, 2),
            visibility_mask.reshape(-1).long(),
        )
        weighted_visibility = multiplier * loss_visibility_weight * loss_visibility
        loss = loss + weighted_visibility
        metrics[f"{prefix}/loss/visibility"] = weighted_visibility

        if valid_tokens.any():
            loss_cmd = F.cross_entropy(
                command_logits[valid_tokens].reshape(-1, self.n_commands),
                target_commands[valid_tokens].reshape(-1).long(),
            )
        else:
            loss_cmd = command_logits.new_zeros(())

        pred_commands = command_logits.argmax(dim=-1)
        command_accuracy = ((pred_commands == target_commands) & valid_tokens).sum().float()
        command_accuracy = command_accuracy / padding_mask.sum().clamp(min=1)

        loss_args = self._continuous_args_loss(args_logits, target_args, target_commands)
        weighted_cmd = multiplier * loss_cmd_weight * loss_cmd
        weighted_args = multiplier * loss_args_weight * loss_args
        loss = loss + weighted_cmd + weighted_args
        metrics[f"{prefix}/loss/cmd"] = weighted_cmd
        metrics[f"{prefix}/loss/args"] = weighted_args
        metrics[f"{prefix}/acc/cmd"] = command_accuracy

        loss_cont, cont_metrics = self._continuity_loss(
            output,
            targets,
            prefix=prefix,
            loss_weight=multiplier * loss_continuity_weight,
        )
        loss = loss + loss_cont
        metrics[f"{prefix}/loss/continuity"] = loss_cont
        metrics.update(cont_metrics)

        loss_align, align_acc = self._alignment_loss(
            output,
            targets,
            prefix=prefix,
            loss_weight=multiplier * loss_alignment_weight,
        )
        loss = loss + loss_align
        metrics[f"{prefix}/loss/alignment"] = loss_align
        metrics[f"{prefix}/acc/alignment"] = align_acc

        loss_consistency = compute_endpoint_consistency_loss(target_commands.detach(), args_logits)
        weighted_consistency = multiplier * loss_consistency_weight * loss_consistency
        loss = loss + weighted_consistency
        metrics[f"{prefix}/loss/consistency"] = weighted_consistency

        pred_points, aux_mask = sample_predicted_auxiliary_points(args_logits, target_commands)
        if aux_mask.any():
            loss_aux = F.mse_loss(pred_points[aux_mask], target_aux_points[aux_mask])
        else:
            loss_aux = args_logits.new_zeros(())
        weighted_aux = multiplier * loss_aux_weight * loss_aux
        loss = loss + weighted_aux
        metrics[f"{prefix}/loss/aux"] = weighted_aux

        return loss, metrics

    def _continuous_args_loss(
        self,
        args_logits: torch.Tensor,
        target_args: torch.Tensor,
        target_commands: torch.Tensor,
    ) -> torch.Tensor:
        pred = args_logits.view(*args_logits.shape[:-1], self.N_ARGS // 2, 2)
        target = target_args.view(*target_args.shape[:-1], self.N_ARGS // 2, 2)
        mask = self.cmd_args_mask[target_commands.long()].view(*target_commands.shape, self.N_ARGS // 2, 2).bool()

        if not mask.any():
            return args_logits.new_zeros(())
        return F.mse_loss(pred[mask], target[mask])

    def _continuity_loss(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        prefix: str,
        loss_weight: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cont_logits = output[f"{prefix}_cont_logits"]
        target = targets["continuity"].long()
        valid_mask = target != -1

        if not valid_mask.any():
            zero = cont_logits.new_zeros(())
            return zero, {f"{prefix}/acc/continuity": cont_logits.new_tensor(1.0)}

        class_weights = cont_logits.new_tensor([1.0, 1.5, 3.0])
        raw_loss = F.cross_entropy(
            cont_logits[valid_mask].reshape(-1, 3),
            target[valid_mask].reshape(-1),
            weight=class_weights,
        )
        pred = cont_logits.argmax(dim=-1)
        accuracy = ((pred == target) & valid_mask).sum().float() / valid_mask.sum().clamp(min=1)
        return loss_weight * raw_loss, {
            f"{prefix}/acc/continuity": accuracy,
        }

    def _alignment_loss(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        prefix: str,
        loss_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        align_logits = output[f"{prefix}_alignment_logits"]
        target = targets["alignment"].long()
        valid_mask = target != -1

        if valid_mask.any():
            raw_loss = F.cross_entropy(
                align_logits[valid_mask].reshape(-1, 3),
                target[valid_mask].reshape(-1),
            )
        else:
            raw_loss = align_logits.new_zeros(())

        pred = align_logits.argmax(dim=-1)
        accuracy = ((pred == target) & valid_mask).sum().float() / valid_mask.sum().clamp(min=1)
        return loss_weight * raw_loss, accuracy

    @staticmethod
    def _single_kl(mu: torch.Tensor, logsigma: torch.Tensor, tolerance: float) -> torch.Tensor:
        loss = -0.5 * torch.mean(1.0 + logsigma - mu.pow(2) - torch.exp(logsigma))
        return loss.clamp(min=tolerance)

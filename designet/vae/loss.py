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


class SVGLoss(nn.Module):
    """VAE training loss for supervised 8-argument SVG reconstruction."""

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
        target_commands = targets["commands"]
        target_args = targets["args"]
        command_logits = output["command_logits"]
        args_logits = output["args_logits"]

        visibility_mask = _get_visibility_mask(target_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(target_commands, seq_dim=-1, extended=True)
        padding_mask = padding_mask * visibility_mask.unsqueeze(-1)

        loss = command_logits.new_zeros(())
        metrics: Dict[str, torch.Tensor] = {}
        loss_cmd_weight = weights["loss_cmd_weight"]
        loss_args_weight = weights["loss_args_weight"]
        loss_visibility_weight = weights["loss_visibility_weight"]
        loss_continuity_weight = weights["loss_continuity_weight"]
        loss_alignment_weight = weights["loss_alignment_weight"]
        loss_consistency_weight = weights["loss_consistency_weight"]
        loss_aux_weight = weights["loss_aux_weight"]

        kl_loss, kl_metrics = self._kl_loss(output, weights)
        loss = loss + kl_loss
        metrics.update(kl_metrics)

        loss_visibility = F.cross_entropy(
            output["visibility_logits"].reshape(-1, 2),
            visibility_mask.reshape(-1).long(),
        )
        weighted_visibility = loss_visibility_weight * loss_visibility
        loss = loss + weighted_visibility
        metrics["loss_visibility"] = weighted_visibility

        target_commands = target_commands[..., 1:]
        target_args = target_args[..., 1:, :]
        padding_mask = padding_mask[..., 1:]
        seq_len = command_logits.size(-2)
        target_commands = target_commands[..., :seq_len]
        target_args = target_args[..., :seq_len, :]
        padding_mask = padding_mask[..., :seq_len]

        cmd_arg_mask = self.cmd_args_mask[target_commands.long()]
        valid_tokens = padding_mask.bool()

        loss_cmd = F.cross_entropy(
            command_logits[valid_tokens].reshape(-1, self.n_commands),
            target_commands[valid_tokens].reshape(-1).long(),
        )

        pred_commands = command_logits.argmax(dim=-1)
        command_accuracy = ((pred_commands == target_commands) & valid_tokens).sum().float()
        command_accuracy = command_accuracy / padding_mask.sum().clamp(min=1)

        loss_args = self._continuous_args_loss(args_logits, target_args, cmd_arg_mask)

        weighted_cmd = loss_cmd_weight * loss_cmd
        weighted_args = loss_args_weight * loss_args
        loss = loss + weighted_cmd
        loss = loss + weighted_args

        metrics["loss_cmd"] = weighted_cmd
        metrics["loss_args"] = weighted_args
        metrics["command_accuracy"] = command_accuracy

        loss_continuity, continuity_metrics = self._continuity_loss(
            output,
            targets,
            loss_continuity_weight,
        )
        loss = loss + loss_continuity
        metrics.update(continuity_metrics)

        loss_alignment, alignment_metrics = self._alignment_loss(
            output,
            targets,
            target_commands,
            loss_alignment_weight,
        )
        loss = loss + loss_alignment
        metrics.update(alignment_metrics)

        consistency_loss = compute_endpoint_consistency_loss(target_commands.detach(), args_logits)
        weighted_consistency = loss_consistency_weight * consistency_loss
        loss = loss + weighted_consistency
        metrics["loss_consistency"] = weighted_consistency

        aux_points = targets["aux_points"][..., 1:, :, :][..., :seq_len, :, :]
        pred_points, aux_mask = sample_predicted_auxiliary_points(args_logits, target_commands)
        if aux_mask.any():
            loss_aux = F.mse_loss(pred_points[aux_mask], aux_points[aux_mask])
        else:
            loss_aux = args_logits.new_zeros(())
        weighted_aux = loss_aux_weight * loss_aux
        loss = loss + weighted_aux
        metrics["loss_aux"] = weighted_aux

        metrics["loss"] = loss
        return metrics

    def _kl_loss(
        self,
        output: Mapping[str, torch.Tensor],
        weights: Mapping[str, float],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        kl_weight = weights["loss_kl_weight"]
        tolerance = weights["kl_tolerance"]

        loss_kl1 = self._single_kl(output["mu1"], output["logsigma1"], tolerance)
        loss_kl2 = self._single_kl(output["mu2"], output["logsigma2"], tolerance)
        return kl_weight * (loss_kl1 + loss_kl2), {
            "loss_kl1": output["mu1"].new_tensor(kl_weight) * loss_kl1,
            "loss_kl2": output["mu2"].new_tensor(kl_weight) * loss_kl2,
        }

    @staticmethod
    def _single_kl(mu: torch.Tensor, logsigma: torch.Tensor, tolerance: float) -> torch.Tensor:
        loss = -0.5 * torch.mean(1.0 + logsigma - mu.pow(2) - torch.exp(logsigma))
        return loss.clamp(min=tolerance)

    def _continuous_args_loss(
        self,
        args_logits: torch.Tensor,
        target_args: torch.Tensor,
        cmd_arg_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = args_logits.view(*args_logits.shape[:-1], self.N_ARGS // 2, 2)
        target = target_args.view(*target_args.shape[:-1], self.N_ARGS // 2, 2)
        mask = cmd_arg_mask.view(*cmd_arg_mask.shape[:-1], self.N_ARGS // 2, 2).bool()

        if not mask.any():
            return args_logits.new_zeros(())
        return F.mse_loss(pred[mask], target[mask])

    def _continuity_loss(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        loss_weight: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cont_logits = output["cont_logits"]
        seq_len = cont_logits.size(-2)
        target_continuity = targets["continuity"][..., 1:][..., :seq_len]
        valid_mask = target_continuity != -1

        if not valid_mask.any():
            zero = cont_logits.new_zeros(())
            return zero, {
                "loss_continuity": zero,
                "continuity_accuracy": cont_logits.new_tensor(1.0),
            }

        class_weights = cont_logits.new_tensor([1.0, 1.5, 3.0])
        loss_raw = F.cross_entropy(
            cont_logits[valid_mask].reshape(-1, 3),
            target_continuity[valid_mask].reshape(-1).long(),
            weight=class_weights,
        )
        weighted_loss = loss_weight * loss_raw

        pred_continuity = cont_logits.argmax(dim=-1)
        correct = (pred_continuity == target_continuity) & valid_mask
        metrics: Dict[str, torch.Tensor] = {
            "loss_continuity": weighted_loss,
            "continuity_accuracy": correct.sum().float() / valid_mask.sum().clamp(min=1),
        }

        return weighted_loss, metrics

    def _alignment_loss(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        target_commands: torch.Tensor,
        loss_weight: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        align_logits = output["alignment_logits"]
        seq_len = align_logits.size(-2)
        target_alignment = targets["alignment"][..., 1:][..., :seq_len]
        line_id = SVGTensor.COMMANDS_SIMPLIFIED.index("l")
        valid_mask = target_commands == line_id

        if valid_mask.any():
            loss_raw = F.cross_entropy(
                align_logits[valid_mask].reshape(-1, 3),
                target_alignment[valid_mask].reshape(-1).long(),
            )
        else:
            loss_raw = align_logits.new_zeros(())

        weighted_loss = loss_weight * loss_raw
        pred_alignment = align_logits.argmax(dim=-1)
        correct = (pred_alignment == target_alignment) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().clamp(min=1)

        return weighted_loss, {
            "loss_alignment": weighted_loss,
            "alignment_accuracy": accuracy,
        }

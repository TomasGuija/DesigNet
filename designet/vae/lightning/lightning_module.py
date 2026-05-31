from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

from designet.checkpoint import normalize_state_dict_keys
from designet.difflib.tensor import SVGTensor
from designet.tensor_utils import ensure_bgs
from designet.vae.diff_refinement import (
    SoftAlignmentRefinerBatched,
    SoftContinuityRefinerBatched,
)
from designet.vae.loss import SVGLoss
from designet.vae.svg_transformer import SVGTransformer


def linear_warmup(step: int, warmup_steps: int, target: float) -> float:
    if warmup_steps <= 0:
        return target
    return target * min(float(step) / float(warmup_steps), 1.0)


STEP_METRICS = {"loss"}
PROGRESS_BAR_METRICS = {"loss"}


class SVGVAELightningModule(L.LightningModule):
    """Lightning module for training the SVG VAE."""

    def __init__(
        self,
        dropout: float = 0.1,
        n_layers: int = 4,
        n_layers_decode: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        d_model: int = 256,
        dim_z: int = 256,
        max_num_groups: int = 4,
        max_seq_len: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        loss_cmd_weight: float = 1.0,
        loss_args_weight: float = 12000.0,
        loss_visibility_weight: float = 1.0,
        loss_continuity_weight: float = 1.0,
        loss_alignment_weight: float = 1.0,
        loss_consistency_weight: float = 5000.0,
        loss_aux_weight: float = 5000.0,
        loss_kl_weight: float = 10.0,
        kl_warmup_steps: int = 10000,
        kl_tolerance: float = 0.1,
        train_with_refinement: bool = False,
        compile_model: bool = False,
        weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_params = {
            "n_commands": len(SVGTensor.COMMANDS_SIMPLIFIED),
            "dropout": dropout,
            "n_layers": n_layers,
            "n_layers_decode": n_layers_decode,
            "n_heads": n_heads,
            "dim_feedforward": dim_feedforward,
            "d_model": d_model,
            "dim_z": dim_z,
            "max_num_groups": max_num_groups,
            "max_seq_len": max_seq_len,
            "num_groups_proposal": max_num_groups,
        }

        self.model = SVGTransformer(self.model_params)
        if weights is not None:
            self._load_model_weights(weights)

        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_cmd_weight = loss_cmd_weight
        self.loss_args_weight = loss_args_weight
        self.loss_visibility_weight = loss_visibility_weight
        self.loss_continuity_weight = loss_continuity_weight
        self.loss_alignment_weight = loss_alignment_weight
        self.loss_consistency_weight = loss_consistency_weight
        self.loss_aux_weight = loss_aux_weight
        self.loss_kl_weight = loss_kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_tolerance = kl_tolerance
        self.train_with_refinement = train_with_refinement
        self.loss_fn = SVGLoss()

        if train_with_refinement:
            self.alignment_refiner = SoftAlignmentRefinerBatched()
            self.continuity_refiner = SoftContinuityRefinerBatched()

    def forward(self, commands: torch.Tensor, args: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(commands=commands, args=args, return_tgt=False)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=15,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        commands = batch["commands"]
        args = batch["args"]
        output = self(commands, args)

        if self.train_with_refinement:
            output = dict(output)
            output["args_logits"] = self._refine_args(output)

        loss_dict = self.loss_fn(output, targets=batch, weights=self._loss_weights())
        self._log_metrics(loss_dict, stage=stage, batch_size=commands.size(0))
        return loss_dict["loss"]

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], *, stage: str, batch_size: int) -> None:
        for name, value in metrics.items():
            log_name = f"{stage}_{name}"
            is_step_metric = stage == "train" and name in STEP_METRICS

            self.log(
                log_name,
                value,
                on_step=is_step_metric,
                on_epoch=True,
                prog_bar=name in PROGRESS_BAR_METRICS,
                logger=True,
                batch_size=batch_size,
            )

    def _refine_args(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_cmds = output["command_logits"].argmax(dim=-1)
        pred_args = output["args_logits"]
        alignment_logits = ensure_bgs(output["alignment_logits"], pred_args, name="alignment_logits")
        cont_logits = ensure_bgs(output["cont_logits"], pred_args, name="cont_logits")

        refined_args = self.alignment_refiner(pred_cmds, pred_args, alignment_logits)
        refined_six_args = self.continuity_refiner(pred_cmds, refined_args[..., -6:], cont_logits)
        return torch.cat(
            [
                refined_args[..., :2],
                refined_six_args[..., :-2],
                refined_args[..., -2:],
            ],
            dim=-1,
        )

    def _loss_weights(self) -> Dict[str, float]:
        return {
            "kl_tolerance": self.kl_tolerance,
            "loss_kl_weight": linear_warmup(self.global_step, self.kl_warmup_steps, self.loss_kl_weight),
            "loss_cmd_weight": self.loss_cmd_weight,
            "loss_args_weight": self.loss_args_weight,
            "loss_visibility_weight": self.loss_visibility_weight,
            "loss_continuity_weight": self.loss_continuity_weight,
            "loss_alignment_weight": self.loss_alignment_weight,
            "loss_consistency_weight": self.loss_consistency_weight,
            "loss_aux_weight": self.loss_aux_weight,
        }

    def _load_model_weights(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = normalize_state_dict_keys(state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing:
            rank_zero_warn(f"Missing keys when loading VAE weights from {checkpoint_path}: {missing}")
        if unexpected:
            rank_zero_warn(f"Unexpected keys when loading VAE weights from {checkpoint_path}: {unexpected}")
        if not missing and not unexpected:
            rank_zero_info(f"Loaded VAE weights from {checkpoint_path} with no missing or unexpected keys.")

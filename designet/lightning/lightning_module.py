from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

from designet.checkpoint import normalize_state_dict_keys
from designet.difflib.tensor import SVGTensor
from designet.lightning.loss import DesigNetLoss
from designet.model import FontConditionalSVGTransformer
from designet.tensor_utils import ensure_bgs
from designet.vae.diff_refinement import (
    SoftAlignmentRefinerBatched,
    SoftContinuityRefinerBatched,
)
from designet.vae.lightning.lightning_module import linear_warmup

STEP_METRICS = {"loss/total"}
PROGRESS_BAR_METRICS = {"loss/total"}


class DesigNetLightningModule(L.LightningModule):
    """Lightning module for training the font-conditional DesigNet generator."""

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
        max_total_len: int = 64,
        encoding_glyphs: Sequence[str] = ("H", "a", "m", "b", "u", "r", "g", "e"),
        decoding_glyphs: Sequence[str] = (
            "A",
            "B",
            "C",
            "c",
            "D",
            "d",
            "E",
            "F",
            "f",
            "G",
            "h",
            "I",
            "i",
            "J",
            "j",
            "K",
            "k",
            "L",
            "l",
            "M",
            "N",
            "n",
            "O",
            "o",
            "P",
            "p",
            "Q",
            "q",
            "R",
            "S",
            "s",
            "T",
            "t",
            "U",
            "V",
            "v",
            "W",
            "w",
            "X",
            "x",
            "Y",
            "y",
            "Z",
            "z",
        ),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        loss_cmd_weight: float = 1.0,
        loss_args_weight: float = 10000.0,
        loss_visibility_weight: float = 1.0,
        loss_continuity_weight: float = 1.0,
        loss_alignment_weight: float = 1.0,
        loss_consistency_weight: float = 5000.0,
        loss_aux_weight: float = 5000.0,
        loss_kl_weight: float = 10.0,
        kl_warmup_steps: int = 10000,
        kl_tolerance: float = 0.1,
        cross_reconstruction_multiplier: float = 1.5,
        train_with_refinement: bool = False,
        compile_model: bool = False,
        vae_checkpoint: str | Path | None = None,
        designet_weights: str | Path | None = None,
        weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if weights is not None:
            rank_zero_warn("`weights` is deprecated; use `designet_weights` instead.")
            if designet_weights is None:
                designet_weights = weights

        self.encoding_glyphs = list(encoding_glyphs)
        self.decoding_glyphs = list(decoding_glyphs)
        self.num_encoding_glyphs = len(self.encoding_glyphs)
        self.num_decoding_glyphs = len(self.decoding_glyphs)
        self.train_with_refinement = train_with_refinement

        self.model_cfg = {
            "n_commands": len(SVGTensor.COMMANDS_SIMPLIFIED),
            "dropout": dropout,
            "n_layers": n_layers,
            "n_layers_decode": n_layers_decode,
            "n_heads": n_heads,
            "dim_feedforward": dim_feedforward,
            "d_model": d_model,
            "dim_z": dim_z,
            "max_num_groups": max_num_groups,
            "num_groups_proposal": max_num_groups,
            "max_seq_len": max_seq_len,
            "max_total_len": max_total_len,
            "encoding_glyphs": self.encoding_glyphs,
            "decoding_glyphs": self.decoding_glyphs,
            "num_encoding_glyphs": self.num_encoding_glyphs,
            "num_decoding_glyphs": self.num_decoding_glyphs,
            "vae_checkpoint": str(vae_checkpoint) if vae_checkpoint is not None else None,
        }

        self.model = FontConditionalSVGTransformer(self.model_cfg)
        if designet_weights is not None:
            self._load_model_weights(designet_weights)

        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.loss_fn = DesigNetLoss()
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
        self.cross_reconstruction_multiplier = cross_reconstruction_multiplier

        if train_with_refinement:
            self.alignment_refiner = SoftAlignmentRefinerBatched()
            self.continuity_refiner = SoftContinuityRefinerBatched()

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(
            input_commands=batch["commands"],
            input_args=batch["args"],
            return_tgt=False,
        )

        if self.train_with_refinement:
            output = dict(output)
            output["self_args_logits"] = self._refine_args(
                commands=output["self_command_logits"],
                args=output["self_args_logits"],
                continuity=output["self_cont_logits"],
                alignment=output["self_alignment_logits"],
            )
            output["cross_args_logits"] = self._refine_args(
                commands=output["cross_command_logits"],
                args=output["cross_args_logits"],
                continuity=output["cross_cont_logits"],
                alignment=output["cross_alignment_logits"],
            )

        return output

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
        batch_size = batch["commands"].size(0)
        output = self(batch)

        loss_dict = self.loss_fn(output, targets=batch, weights=self._loss_weights())
        self._log_metrics(loss_dict, stage=stage, batch_size=batch_size)
        return loss_dict["loss/total"]

    def _refine_args(
        self,
        *,
        commands: torch.Tensor,
        args: torch.Tensor,
        continuity: torch.Tensor,
        alignment: torch.Tensor,
    ) -> torch.Tensor:
        pred_cmds = commands.argmax(dim=-1)
        alignment = ensure_bgs(alignment, args, name="alignment_logits")
        continuity = ensure_bgs(continuity, args, name="cont_logits")

        refined_args = self.alignment_refiner(pred_cmds, args, alignment)
        refined_six_args = self.continuity_refiner(pred_cmds, refined_args[..., -6:], continuity)
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
            "cross_reconstruction_multiplier": self.cross_reconstruction_multiplier,
        }

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], *, stage: str, batch_size: int) -> None:
        for name, value in metrics.items():
            is_step_metric = stage == "train" and name in STEP_METRICS
            self.log(
                f"{stage}/{name}",
                value,
                on_step=is_step_metric,
                on_epoch=True,
                prog_bar=name in PROGRESS_BAR_METRICS,
                logger=True,
                batch_size=batch_size,
            )

            if name == "loss/total":
                self.log(
                    f"{stage}_loss",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=False,
                    batch_size=batch_size,
                )

    def _load_model_weights(self, weights_path: str | Path) -> None:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = normalize_state_dict_keys(state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing:
            rank_zero_warn(f"Missing keys when loading DesigNet weights from {weights_path}: {missing}")
        if unexpected:
            rank_zero_warn(f"Unexpected keys when loading DesigNet weights from {weights_path}: {unexpected}")
        if not missing and not unexpected:
            rank_zero_info(f"Loaded DesigNet weights from {weights_path} with no missing or unexpected keys.")

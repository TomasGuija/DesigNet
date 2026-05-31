from __future__ import annotations

import torch
from lightning.pytorch.cli import LightningCLI

from designet.lightning.config_validation import ensure_matching_config
from designet.vae.lightning.data_module import GlyphDataModule
from designet.vae.lightning.lightning_module import SVGVAELightningModule


class SVGVAELightningCLI(LightningCLI):
    def before_fit(self) -> None:
        ensure_matching_config(
            model_config=self.model.model_params,
            data_config=self.datamodule.hparams,
            pairs=(
                ("max_num_groups", "max_num_groups"),
                ("max_seq_len", "max_seq_len"),
            ),
        )


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    SVGVAELightningCLI(
        SVGVAELightningModule,
        GlyphDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()

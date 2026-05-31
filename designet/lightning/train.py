from __future__ import annotations

import torch
from lightning.pytorch.cli import LightningCLI

from designet.lightning.config_validation import ensure_matching_config
from designet.lightning.data_module import FontDataModule
from designet.lightning.lightning_module import DesigNetLightningModule


class DesigNetLightningCLI(LightningCLI):
    def before_fit(self) -> None:
        ensure_matching_config(
            model_config=self.model.model_cfg,
            data_config=self.datamodule.hparams,
            pairs=(
                ("max_num_groups", "max_num_groups"),
                ("max_seq_len", "max_seq_len"),
                ("max_total_len", "max_total_len"),
                ("encoding_glyphs", "encoding_letters"),
                ("decoding_glyphs", "decoding_letters"),
            ),
        )


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    DesigNetLightningCLI(
        DesigNetLightningModule,
        FontDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()

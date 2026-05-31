from __future__ import annotations

import torch
from lightning.pytorch.cli import LightningCLI

from designet.vae.lightning.data_module import GlyphDataModule
from designet.vae.lightning.lightning_module import SVGVAELightningModule


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    LightningCLI(
        SVGVAELightningModule,
        GlyphDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()

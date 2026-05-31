<h1 align="center">DesigNet: Learning to Draw Vector Graphics as Designers Do</h1>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomasGuija/DesigNet/blob/main/demo_designet_inference.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2604.06494-b31b1b.svg)](https://arxiv.org/abs/2604.06494)

<p align="center">
  <img src="docs/imgs/overview.jpg" width="500"/>
</p>

# Introduction

This is the official code release for the paper **"DesigNet: Learning to Draw Vector Graphics as Designers Do"**. The repository includes inference, evaluation, and training code for both self-reconstruction with our variational autoencoder and full font generation from a subset of reference characters.

Part of the code found here was inspired by [DeepSVG](https://github.com/alexandre01/deepsvg). The base model builds on their Transformer-based autoencoder with incremental improvements, and a simplified version of their Deep Learning SVG Library is included.

# Main contributions

This codebase includes:

* An SVG variational autoencoder.
* A font generative model for full font reconstruction from a subset of reference characters.
* Continuity and alignment self-refinement modules for producing more accurate editable vector outputs while using continuous coordinates.
* Code to load glyphs directly from SVG data, train and fine-tune models, run inference, evaluate pretrained models, and visualize generated SVG outputs.


<p align="center">
  <img src="docs/imgs/gif.gif" width="750"/>
</p>

<p align="center">
  <em>Visual summary of DesigNet: from reference glyphs to editable SVG font generation.</em>
</p>

# Updates

* **April 2026:** First public release with inference code and usage guidelines.

* **May 2026:** Public dataset and evaluation scripts added. The evaluation code computes Chamfer reconstruction error, rendered-image IoU, rendered-image L1 distance, and optional continuity/alignment accuracy. Missing checkpoints and datasets are downloaded automatically from Hugging Face 🤗.

* **May 2026:** Training code added using PyTorch Lightning, together with default training configurations and the loss components used for supervision.

# Installation

```bash
pip install -e .
```

# Training

Training is provided through a PyTorch Lightning based interface for both the [`variational autoencoder`](./designet/vae/lightning) and the [`font generative model`](./designet/lightning). Default training configurations are available in [`config`](config/) as `.yaml` files.

Before launching training, update the corresponding configuration file if needed, especially the dataset paths and split files.

## Training from scratch

To train the font generative model from scratch, run:

```bash
python designet/lightning/train.py fit --config config/designet.yaml
```

To train the variational autoencoder from scratch, run:

```bash
python designet/vae/lightning/train.py fit --config config/vae.yaml
```

The training entry points use LightningCLI, so most trainer, model, and data arguments can also be overridden from the command line. For example:

```bash
python designet/lightning/train.py fit \
  --config config/designet.yaml \
  --trainer.max_epochs 100 \
  --data.batch_size 16
```

## Resuming an interrupted training run

To resume an interrupted training run produced with the same code and configuration, use the Lightning `--ckpt_path` parameter:

```bash
python designet/lightning/train.py fit \
  --config config/designet.yaml \
  --ckpt_path path/to/last.ckpt
```

or, for the variational autoencoder:

```bash
python designet/vae/lightning/train.py fit \
  --config config/vae.yaml \
  --ckpt_path path/to/last.ckpt
```

This restores the full Lightning training state, including model weights, optimizer state, scheduler state, epoch/step counters, and saved hyperparameters.

> **Checkpoint compatibility note:** `--ckpt_path` is intended for resuming training runs generated with the same training code and configuration structure. Some released pretrained checkpoints may contain legacy configuration fields or nested dictionary arguments saved in their Lightning hyperparameters. In those cases, directly passing them through `--ckpt_path` may fail because the current LightningCLI parser no longer exposes the same internal argument structure. This does not affect inference or weight initialization; it only concerns full Lightning training-state restoration.

## Initializing from pretrained weights

If the goal is to initialize a model from pretrained weights rather than resume the exact original training state, use the model-specific weight-loading arguments instead of `--ckpt_path`.

For the font generative model, use `designet_weights` to initialize from pretrained DesigNet weights:

```bash
python designet/lightning/train.py fit \
  --config config/designet.yaml \
  --model.designet_weights path/to/DesigNet.ckpt
```

To train the font generative model from a pretrained variational autoencoder, use `vae_checkpoint`:

```bash
python designet/lightning/train.py fit \
  --config config/designet.yaml \
  --model.vae_checkpoint path/to/VAE.ckpt
```

For the variational autoencoder, use `weights` to initialize from pretrained VAE weights:

```bash
python designet/vae/lightning/train.py fit \
  --config config/vae.yaml \
  --model.weights path/to/VAE.ckpt
```

These options load the relevant model parameters without restoring the full Lightning trainer state, making them more appropriate for fine-tuning, transfer learning, or restarting training from released checkpoints.

# Inference

To run a pretrained checkpoint, an inference interface is provided both for the [`variational autoencoder`](./designet/vae/tools.py) and for the [`font generative model`](./designet/tools.py). Shared SVG and tensor utilities live in [`designet/svg_utils.py`](./designet/svg_utils.py), [`designet/tensor_utils.py`](./designet/tensor_utils.py), and [`designet/geometry.py`](./designet/geometry.py). For a usage guide, you may run the demo notebooks, which include:

* Downloading our pretrained checkpoints from Hugging Face.
* Both self and cross reconstruction.
* Visualizing outputs and exporting them to SVG format.
* Latent space interpolation.
* Applying our self-refinement modules.

# Dataset

The public SVG dataset is hosted on Hugging Face at [`TomasGuija/LatinFontsSVGs`](https://huggingface.co/datasets/TomasGuija/LatinFontsSVGs). If `--data_dir` is omitted, the evaluation scripts download and extract it under `data/LatinFontsSVGs`. If `--csv_path` is omitted, they use `data/test.csv` from the downloaded dataset repository.

For training, make sure that the dataset directory and split CSV files referenced in the corresponding configuration file point to the desired local or downloaded data.

# Evaluation

Evaluate the VAE:

```bash
python -m designet.eval.evaluate_vae
```

Evaluate DesigNet self- and cross-reconstruction:

```bash
python -m designet.eval.evaluate_designet
```

Both scripts accept `--model_ckpt`, `--data_dir`, `--csv_path`, `--device`, `--batch_size`, `--max_batches`, and `--output_json`. Optional flags `--eval_continuity` and `--eval_alignment` enable the geometry-derived constraint metrics.

# Citation
If you use this code, dataset, or pretrained models in your research, please cite our work:

```bibtex
@article{GUIJAVALIENTE2026104627,
  title = {DesigNet: Learning to draw vector graphics as designers do},
  journal = {Computers & Graphics},
  volume = {137},
  pages = {104627},
  year = {2026},
  issn = {0097-8493},
  doi = {https://doi.org/10.1016/j.cag.2026.104627},
  url = {https://www.sciencedirect.com/science/article/pii/S0097849326000981},
  author = {Tomas Guija-Valiente and Iago Suárez}
}
```

# Acknowledgements

[*DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation*](https://arxiv.org/abs/2007.11301)
Alexandre Carlier, Martin Danelljan, Alexandre Alahi, Radu Timofte
*CoRR*, 2020

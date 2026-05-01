<h1 align="center">DesigNet: Learning to Draw Vector Graphics as Designers Do</h1>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomasGuija/DesigNet/blob/main/demo_designet_inference.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2604.06494-b31b1b.svg)](https://arxiv.org/abs/2604.06494)

<p align="center">
  <img src="docs/imgs/overview.jpg" width="500"/>
</p>


# Introduction

This is the official code for the paper "DesigNet:  Learning to Draw Vector Graphics as Designers Do". It includes inference and evaluation code, both for self-reconstruction with our variational autoencoder and for full font generation from a subset of reference characters.

Part of the code found here was inspired by the work from [DeepSVG](https://github.com/alexandre01/deepsvg). The base model is built on top of their Transformer-based autoencoder with incremental improvements, and a simplified version of their Deep Learning SVG Library is included.

# Main contributions

This codebase includes:
* A SVG variational autoencoder.
* A Font Generative Model for full font reconstruction from a subset of reference characters.
* Continuity and alignment self-refinement modules for providing more accurate editable outputs while using continuous coordinates.
* Code to load glyphs directly from SVG data, run inference, evaluate pretrained models, and visualize results.


# Updates

* April 2026: First commit with inference code and guidelines.

* May 2026: Public dataset and evaluation scripts added. The evaluation code computes Chamfer reconstruction error, rendered-image IoU, rendered-image L1 distance, and optional continuity/alignment accuracy. Missing checkpoints and datasets are downloaded automatically from Hugging Face 🤗.

# Installation

```
pip install -e .
```

# Inference

To run a pretrained checkpoint, an inference interface is provided both for the [`variational autoencoder`](./designet/vae/tools.py) and for the [`font generative model`](./designet/tools.py). Shared SVG and tensor utilities live in [`designet/svg_utils.py`](./designet/svg_utils.py), [`designet/tensor_utils.py`](./designet/tensor_utils.py), and [`designet/geometry.py`](./designet/geometry.py). For a usage guide, you may run the demo notebooks, which include:

* Downloading our pretrained checkpoints from Hugging Face.
* Both self and cross reconstruction.
* Visualizing outputs and exporting them to SVG format.
* Latent space interpolation.
* Applying our self-refinement modules.

# Dataset

The public SVG dataset is hosted on Hugging Face at [`TomasGuija/LatinFontsSVGs`](https://huggingface.co/datasets/TomasGuija/LatinFontsSVGs). If `--data_dir` is omitted, the evaluation scripts download and extract it under `data/LatinFontsSVGs`. If `--csv_path` is omitted, they use `data/test.csv` from the downloaded dataset repository.

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

```bibtex
@misc{guijavaliente2026designetlearningdrawvector,
      title={DesigNet: Learning to Draw Vector Graphics as Designers Do},
      author={Tomas Guija-Valiente and Iago Suárez},
      year={2026},
      eprint={2604.06494},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.06494},
}
```

# Acknowledgements

[*DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation*](https://arxiv.org/abs/2007.11301)
Alexandre Carlier, Martin Danelljan, Alexandre Alahi, Radu Timofte
*CoRR*, 2020

<h1 align="center">DesigNet: Learning to Draw Vector Graphics as Designers Do</h1>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomasGuija/DesigNet/blob/main/demo_designet_inference.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2604.06494-b31b1b.svg)](https://arxiv.org/abs/2604.06494)

<p align="center">
  <img src="docs/imgs/overview.jpg" width="500"/>
</p>


# Introduction

This is the official code for the paper "DesigNet:  Learning to Draw Vector Graphics as Designers Do". It includes the inference code, both for self-reconstruction with our variational autoencoder, and for full font generation from a subset of reference characters. We will provide the evaluation and training code soon.

Part of the code found here was inspired by the work from [DeepSVG](https://github.com/alexandre01/deepsvg). The base model is built on top of their Transformer-based autoencoder with incremental improvements, and a simplified version of their Deep Learning SVG Library is included.

# Main contributions

This codebase includes:
* A SVG variational autoencoder.
* A Font Generative Model for full font reconstruction from a subset of reference characters.
* Continuity and alignment self-refinement modules for providing more accurate editable outputs while using continuous coordinates.
* All necessary code to load glyhps directly from SVG data, run inference and visualize results.


# Updates

* April 2026: First commit with inference code and guidelines.


# Installation

```
pip install -e .
```

# Inference

To run a pretrained checkpoint, an inference interface is provided both for the [`variational autoencoder`](./designet/vae/tools.py) and for the [`font generative model`](./designet/tools.py). For a usage guide, you may run the demo notebooks, which include:

* Downloading our pretrained checkpoints.
* Both self and cross reconstruction.
* Visualizing outputs and exporting them to SVG format.
* Latent space interpolation.
* Applying our self-refinement modules.

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

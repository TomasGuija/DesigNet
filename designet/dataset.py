from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

HF_DATASET_REPO_ID = "TomasGuija/LatinFontsSVGs"
HF_DATASET_ARCHIVE = "LatinFontsSVGs.zip"
DEFAULT_DATA_ROOT = Path("data")
DATASET_DIR_NAME = "LatinFontsSVGs"
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / DATASET_DIR_NAME
DEFAULT_REQUIRED_GB = 5.0


def has_enough_disk_space(path: str | Path, required_gb: float) -> bool:
    """Return True if ``path`` has at least ``required_gb`` free disk space."""
    free_bytes = shutil.disk_usage(path).free
    return free_bytes >= int(required_gb * 1024**3)


def resolve_dataset_path(
    path: str | Path | None = None,
    required_gb: float = DEFAULT_REQUIRED_GB,
) -> str:
    """
    Return the local SVG dataset directory.

    ``path`` must point to the SVG dataset directory itself, e.g.
    ``data/LatinFontsSVGs``. When omitted, the public dataset is downloaded
    under ``data/LatinFontsSVGs`` if it is not already present.
    """
    dataset_dir = Path(path) if path is not None else DEFAULT_DATASET_DIR

    if dataset_dir.exists():
        if _looks_like_svg_dataset_dir(dataset_dir):
            return str(dataset_dir)
        raise ValueError(
            f"Dataset directory does not look like an SVG dataset root: {dataset_dir}. "
            "Expected a directory containing paths like family_*/font_*/*.svg."
        )

    if path is not None:
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_root = dataset_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)

    if not has_enough_disk_space(data_root, required_gb):
        free_gb = shutil.disk_usage(data_root).free / 1024**3
        raise RuntimeError(
            f"Not enough disk space. Required: {required_gb:.1f} GB, " f"available: {free_gb:.1f} GB at {data_root}."
        )

    print(f"Downloading dataset repo from Hugging Face: {HF_DATASET_REPO_ID}")

    snapshot_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=data_root,
        local_dir_use_symlinks=False,
    )

    archive_path = data_root / HF_DATASET_ARCHIVE
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    print(f"Extracting {archive_path} into {data_root}")

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(data_root)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found after extraction: {dataset_dir}")

    print(f"Dataset ready at {dataset_dir}")
    return str(dataset_dir)


def _looks_like_svg_dataset_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("family_*/font_*/*.svg"))

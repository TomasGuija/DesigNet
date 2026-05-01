from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

HF_DATASET_REPO_ID = "TomasGuija/LatinFontsSVGs"
HF_DATASET_ARCHIVE = "LatinFontsSVGs.zip"
DEFAULT_DATA_ROOT = Path("data")
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
    """
    data_root = Path(path) if path is not None else DEFAULT_DATA_ROOT
    dataset_dir = data_root / "LatinFontsSVGs"

    if dataset_dir.exists():
        return str(dataset_dir)

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

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import hf_hub_download

HF_REPO_ID = "TomasGuija/DesigNet"
_HF_DESIGNET_FILE = "DesigNet.ckpt"
_HF_VAE_FILE = "VAE.ckpt"


def resolve_checkpoint_path(path: str, hf_filename: str = _HF_DESIGNET_FILE) -> str:
    """Return a local file path, downloading (and extracting if zipped) from HF Hub if needed."""
    if Path(path).exists():
        return path
    print(f"Downloading {hf_filename} from HuggingFace Hub ({path})...")
    local_path = Path(hf_hub_download(repo_id=path, filename=hf_filename))
    print("Download complete.")
    return str(local_path)


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrapper prefixes (model., net., module., _orig_mod.) from state dict keys."""
    prefixes = ("model.", "net.", "module.", "_orig_mod.")
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out

#!/usr/bin/env python3
"""
Evaluate a pretrained SVG VAE on a glyph dataset.

The script expects two dataset inputs:
    1. ``--data_dir``: root directory containing SVG glyph files.
    2. ``--csv_path``: CSV file describing which glyphs should be evaluated.

Evaluation computes glyph-level reconstruction metrics:
    - Chamfer reconstruction error (RE)
    - pixel Intersection over Union (IoU)
    - pixel L1 distance
    - optional geometry-derived continuity and line-alignment accuracy
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from designet.dataset import resolve_dataset_path
from designet.eval.common import (
    MetricResults,
    default_test_csv,
    empty_metric_results,
    evaluate_svg_batch,
    extend_results,
    move_batch_to_device,
    print_metric_summary,
    resolve_device,
    save_summary,
    summarize_values,
)
from designet.vae.glyph_dataset import GlyphDataset
from designet.vae.tools import load_vae_model, refine_output_with_soft_refinement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained SVG VAE on a glyph dataset.")

    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="VAE checkpoint path. If not found, the default VAE checkpoint is downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset directory. If missing or not found, the dataset is downloaded automatically.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="CSV metadata file. If missing, defaults to <data_dir>/test.csv.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional maximum number of batches to evaluate.",
    )
    parser.add_argument(
        "--use_self_refine",
        action="store_true",
        help="Apply soft self-refinement before greedy decoding.",
    )
    parser.add_argument(
        "--eval_continuity",
        action="store_true",
        help="Evaluate geometry-derived continuity accuracy.",
    )
    parser.add_argument(
        "--eval_alignment",
        action="store_true",
        help="Evaluate geometry-derived line-alignment accuracy.",
    )
    parser.add_argument(
        "--no_re",
        action="store_true",
        help="Disable Chamfer reconstruction error evaluation.",
    )
    parser.add_argument(
        "--no_iou",
        action="store_true",
        help="Disable pixel IoU evaluation.",
    )
    parser.add_argument(
        "--no_l1",
        action="store_true",
        help="Disable pixel L1 evaluation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path where metric summary will be saved as JSON.",
    )

    return parser.parse_args()


def build_eval_dataloader(
    cfg: Dict,
    data_dir: str,
    csv_path: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = GlyphDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        max_num_groups=cfg["max_num_groups"],
        max_seq_len=cfg["max_seq_len"],
        max_total_len=cfg["max_total_len"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=GlyphDataset.collate_fn,
    )


@torch.no_grad()
def decode_batch(
    model,
    batch: Dict,
    use_self_refine: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = model(
        commands=batch["commands"],
        args=batch["args"],
        return_tgt=False,
    )

    if use_self_refine:
        pred = refine_output_with_soft_refinement(pred)

    pred_cmds, pred_args = model.greedy_sample(
        pred=pred,
        close_paths=True,
    )

    return pred_cmds, pred_args


@torch.no_grad()
def evaluate_vae(
    model,
    data_loader: Iterable[Dict],
    device: torch.device,
    eval_re: bool = True,
    eval_iou: bool = True,
    eval_l1: bool = True,
    use_self_refine: bool = False,
    eval_continuity: bool = False,
    eval_alignment: bool = False,
) -> MetricResults:
    """
    Evaluate VAE glyph reconstructions on a dataloader.
    """
    model = model.eval().to(device)
    all_results = empty_metric_results()

    for batch in tqdm(data_loader, desc="Evaluating VAE"):
        batch = move_batch_to_device(batch, device)

        pred_cmds, pred_args = decode_batch(
            model=model,
            batch=batch,
            use_self_refine=use_self_refine,
        )

        batch_results = evaluate_svg_batch(
            gt_cmds=batch["commands"],
            gt_args=batch["args"],
            pred_cmds=pred_cmds,
            pred_args=pred_args,
            eval_re=eval_re,
            eval_iou=eval_iou,
            eval_l1=eval_l1,
            eval_continuity=eval_continuity,
            eval_alignment=eval_alignment,
            gt_continuity=batch.get("continuity"),
            gt_alignment=batch.get("alignment"),
        )
        extend_results(all_results, batch_results)

    return all_results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, cfg, _ = load_vae_model(
        checkpoint=args.model_ckpt,
        device=device,
    )

    data_dir = Path(resolve_dataset_path(args.data_dir))
    csv_path = Path(args.csv_path) if args.csv_path is not None else default_test_csv(data_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data_loader = build_eval_dataloader(
        cfg=cfg,
        data_dir=str(data_dir),
        csv_path=str(csv_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.max_batches is not None:
        data_loader = islice(data_loader, args.max_batches)

    results = evaluate_vae(
        model=model,
        data_loader=data_loader,
        device=device,
        eval_re=not args.no_re,
        eval_iou=not args.no_iou,
        eval_l1=not args.no_l1,
        use_self_refine=args.use_self_refine,
        eval_continuity=args.eval_continuity,
        eval_alignment=args.eval_alignment,
    )

    summary = {key: summarize_values(values) for key, values in results.items()}
    print_metric_summary(summary)

    if args.output_json is not None:
        output_path = save_summary(summary, args.output_json)
        print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()

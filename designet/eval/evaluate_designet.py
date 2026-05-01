#!/usr/bin/env python3
"""
Evaluate a pretrained DesigNet generative model on a font dataset.

Evaluation computes:
    - self-reconstruction metrics on encoding glyphs
    - cross-reconstruction metrics on decoding glyphs
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
from designet.font_dataset import FontDataset
from designet.tools import load_designet_model, refine_output_with_soft_refinement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained DesigNet model on a font dataset.")

    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="DesigNet checkpoint path. If not found or None, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset directory. If missing or None, downloads automatically.",
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
        default=8,
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
    dataset = FontDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        max_num_groups=cfg["max_num_groups"],
        max_seq_len=cfg["max_seq_len"],
        max_total_len=cfg.get("max_total_len"),
        encoding_letters=cfg["encoding_glyphs"],
        decoding_letters=cfg["decoding_glyphs"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=FontDataset.collate_fn,
    )


@torch.no_grad()
def decode_batch(
    model,
    batch: Dict,
    use_self_refine: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = model(
        input_commands=batch["commands"],
        input_args=batch["args"],
        return_tgt=False,
    )

    if use_self_refine:
        pred = refine_output_with_soft_refinement(pred, align_min_conf=0.75, cont_min_conf=0.5)

    pred_cmds_cross, pred_args_cross, pred_cmds_self, pred_args_self = model.greedy_sample(pred=pred)

    return pred_cmds_self, pred_args_self, pred_cmds_cross, pred_args_cross


def flatten_targets(
    commands: torch.Tensor,
    args: torch.Tensor,
    num_encoding_glyphs: int,
    num_decoding_glyphs: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, num_groups, seq_len = commands.shape

    self_cmds = commands[:, :num_encoding_glyphs]
    self_args = args[:, :num_encoding_glyphs]

    cross_cmds = commands[:, num_encoding_glyphs : num_encoding_glyphs + num_decoding_glyphs]
    cross_args = args[:, num_encoding_glyphs : num_encoding_glyphs + num_decoding_glyphs]

    self_cmds = self_cmds.reshape(-1, num_groups, seq_len)
    self_args = self_args.reshape(-1, num_groups, seq_len, 8)

    cross_cmds = cross_cmds.reshape(-1, num_groups, seq_len)
    cross_args = cross_args.reshape(-1, num_groups, seq_len, 8)

    self_args = self_args[..., 2:]
    cross_args = cross_args[..., 2:]

    return self_cmds, self_args, cross_cmds, cross_args


def print_designet_summary(summary: Dict[str, Dict]) -> None:
    for split_name in ["self", "cross"]:
        print()
        print(f"{split_name.upper()} reconstruction")
        print_metric_summary(summary[split_name])


@torch.no_grad()
def evaluate_designet(
    model,
    data_loader: Iterable[Dict],
    device: torch.device,
    eval_re: bool = True,
    eval_iou: bool = True,
    eval_l1: bool = True,
    use_self_refine: bool = False,
    eval_continuity: bool = False,
    eval_alignment: bool = False,
) -> Dict[str, MetricResults]:
    """
    Evaluate DesigNet self and cross reconstructions on a dataloader.

    Optional continuity/alignment metrics compare labels recomputed from decoded
    SVG geometry against labels recomputed from ground-truth geometry.
    """
    model = model.eval().to(device)

    results = {"self": empty_metric_results(), "cross": empty_metric_results()}

    for batch in tqdm(data_loader, desc="Evaluating DesigNet"):
        batch = move_batch_to_device(batch, device)

        pred_cmds_self, pred_args_self, pred_cmds_cross, pred_args_cross = decode_batch(
            model=model,
            batch=batch,
            use_self_refine=use_self_refine,
        )

        self_cmds, self_args, cross_cmds, cross_args = flatten_targets(
            commands=batch["commands"],
            args=batch["args"],
            num_encoding_glyphs=model.num_encoding_letters,
            num_decoding_glyphs=model.num_decoding_letters,
        )

        self_results = evaluate_svg_batch(
            gt_cmds=self_cmds,
            gt_args=self_args,
            pred_cmds=pred_cmds_self,
            pred_args=pred_args_self,
            eval_re=eval_re,
            eval_iou=eval_iou,
            eval_l1=eval_l1,
            eval_continuity=eval_continuity,
            eval_alignment=eval_alignment,
        )

        cross_results = evaluate_svg_batch(
            gt_cmds=cross_cmds,
            gt_args=cross_args,
            pred_cmds=pred_cmds_cross,
            pred_args=pred_args_cross,
            eval_re=eval_re,
            eval_iou=eval_iou,
            eval_l1=eval_l1,
            eval_continuity=eval_continuity,
            eval_alignment=eval_alignment,
        )

        extend_results(results["self"], self_results)
        extend_results(results["cross"], cross_results)

    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, cfg, _ = load_designet_model(
        checkpoint=args.model_ckpt,
        device=device,
        strict=False,
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

    results = evaluate_designet(
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

    summary = {
        mode: {metric: summarize_values(values) for metric, values in mode_results.items()}
        for mode, mode_results in results.items()
    }

    print_designet_summary(summary)

    if args.output_json is not None:
        output_path = save_summary(summary, args.output_json)
        print(f"\nSaved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()

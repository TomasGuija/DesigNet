from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from designet.difflib.tensor import SVGTensor
from designet.eval.losses import (
    compute_iou,
    compute_l1,
    geometric_constraint_correctness,
    reconstruction_error,
)
from designet.svg_utils import svg_from_cmd_args
from designet.svglib.geom import Bbox
from designet.svglib.svg import SVG

MetricResults = Dict[str, List[float]]
MetricSummary = Dict[str, Optional[float]]

METRIC_NAMES = {
    "re": "RE",
    "iou": "IoU",
    "l1": "L1",
    "continuity": "Continuity accuracy",
    "alignment": "Alignment accuracy",
}
ACCURACY_METRICS = {"continuity", "alignment"}


def empty_metric_results() -> MetricResults:
    return {metric: [] for metric in METRIC_NAMES}


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def default_test_csv(data_dir: Path) -> Path:
    return data_dir.parent / "test.csv"


def extend_results(dst: MetricResults, src: MetricResults) -> None:
    for key, values in src.items():
        dst[key].extend(values)


def summarize_values(values: List[float]) -> MetricSummary:
    if not values:
        return {"count": 0, "mean": None, "std": None}

    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item() if len(values) > 1 else 0.0,
    }


def save_summary(summary: Dict, output_json: str) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return output_path


def print_metric_summary(summary: Dict[str, MetricSummary]) -> None:
    for metric, display_name in METRIC_NAMES.items():
        item = summary[metric]
        if item["count"] == 0:
            print(f"{display_name}: skipped")
        elif metric in ACCURACY_METRICS:
            correct = round(item["mean"] * item["count"])
            print(f"{display_name}: {item['mean']:.6f} ({correct}/{item['count']} valid positions)")
        else:
            print(f"{display_name}: {item['mean']:.6f} +/- {item['std']:.6f} ({item['count']} samples)")


def _target_svg_from_cmd_args(commands: torch.Tensor, args: torch.Tensor) -> SVG:
    valid = (commands != 4) & (commands != 3)
    tensor = SVGTensor.from_cmd_args(
        commands[valid].cpu(),
        args[valid].cpu()[..., -6:],
    )
    return SVG.from_tensor(tensor.data, viewbox=Bbox(64), allow_empty=True).split_paths()


def evaluate_svg_batch(
    gt_cmds: torch.Tensor,
    gt_args: torch.Tensor,
    pred_cmds: torch.Tensor,
    pred_args: torch.Tensor,
    *,
    eval_re: bool = True,
    eval_iou: bool = True,
    eval_l1: bool = True,
    eval_continuity: bool = False,
    eval_alignment: bool = False,
) -> MetricResults:
    """
    Compute glyph-level metrics for a decoded SVG batch.

    Continuity and alignment are recomputed from decoded geometry and scored at
    valid ground-truth constraint positions.
    """
    results = empty_metric_results()

    if eval_continuity:
        results["continuity"].extend(
            geometric_constraint_correctness(
                gt_cmds=gt_cmds,
                gt_args=gt_args,
                pred_cmds=pred_cmds,
                pred_args=pred_args,
                constraint="continuity",
            )
        )

    if eval_alignment:
        results["alignment"].extend(
            geometric_constraint_correctness(
                gt_cmds=gt_cmds,
                gt_args=gt_args,
                pred_cmds=pred_cmds,
                pred_args=pred_args,
                constraint="alignment",
            )
        )

    for sample_idx in range(gt_cmds.size(0)):
        gt_svg = _target_svg_from_cmd_args(gt_cmds[sample_idx], gt_args[sample_idx])
        pred_svg = svg_from_cmd_args(pred_cmds[sample_idx], pred_args[sample_idx])

        if pred_svg is None:
            print("Warning: predicted SVG is empty or invalid. Skipping sample.")
            continue

        gt_svg.numericalize(n=256, round_coords=False)
        pred_svg.numericalize(n=256, round_coords=False)

        if eval_re:
            results["re"].append(float(reconstruction_error(gt_svg, pred_svg)))
        if eval_iou:
            results["iou"].append(float(compute_iou(pred_svg, gt_svg)))
        if eval_l1:
            results["l1"].append(float(compute_l1(pred_svg, gt_svg)))

    return results

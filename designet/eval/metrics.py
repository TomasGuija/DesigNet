import numpy as np
import torch

from designet.geometry import compute_continuity_tensor, compute_line_alignment_tensor


def svg_to_png(svg_obj, size=64):
    import io

    import cairosvg
    from PIL import Image

    svg_str = svg_obj.to_str_single_path(fill=True, color="black", size=size)

    png_bytes = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=size,
        output_height=size,
    )

    img_rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Composite transparent pixels over white
    white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img_rgba)

    return img.convert("L")


def chamfer_loss_vectorized(gt_points: torch.Tensor, pred_points: torch.Tensor):
    """
    Compute symmetric Chamfer distance between two point sets.

    Args:
        gt_points (Tensor): Ground-truth points of shape (N, 2).
        pred_points (Tensor): Predicted points of shape (M, 2).
    Returns:
        Tensor: Scalar Chamfer distance.
    """
    if gt_points.numel() == 0 or pred_points.numel() == 0:
        return gt_points.new_tensor(torch.nan)

    dist_matrix = torch.cdist(gt_points, pred_points)
    min1 = dist_matrix.min(dim=-1).values.mean(dim=-1)
    min2 = dist_matrix.min(dim=-2).values.mean(dim=-1)
    chamfer = 0.5 * (min1 + min2)

    return chamfer


def resample_fixed_points(points, fixed_num_points=20):
    """
    Uniformly resample a point sequence to a fixed number of points.

    Args:
        points (Tensor): Input points of shape (N, 2).
        fixed_num_points (int): Number of sampled output points.

    Returns:
        Tensor: Resampled points.
    """
    N = points.shape[0]

    if N == 0:
        return points

    idxs = torch.linspace(0, N - 1, steps=fixed_num_points).long()
    return points[idxs]


def reconstruction_error(svg_gt, svg_pred):
    """
    Compute Chamfer reconstruction error between two SVG objects.

    Args:
        svg_gt: Ground-truth SVG object.
        svg_pred: Predicted SVG object.
        norm (bool): Whether to normalize coordinates.

    Returns:
        Tensor: Chamfer distance or NaN if invalid.
    """
    if svg_gt.empty() or svg_pred.empty():
        return torch.tensor(torch.nan, dtype=torch.float32)

    sampled_pred_tensor = [
        resample_fixed_points(torch.tensor(path[0].sample_points(), dtype=torch.float32))
        for path in svg_pred.svg_path_groups
    ]
    sampled_pred_tensor = [t for t in sampled_pred_tensor if len(t) > 0]

    sampled_gt_tensor = [
        resample_fixed_points(torch.tensor(path[0].sample_points(), dtype=torch.float32))
        for path in svg_gt.svg_path_groups
    ]
    sampled_gt_tensor = [t for t in sampled_gt_tensor if len(t) > 0]

    if len(sampled_gt_tensor) == 0 or len(sampled_pred_tensor) == 0:
        return torch.tensor(torch.nan, dtype=torch.float32)

    sampled_gt_tensor = torch.cat(sampled_gt_tensor, dim=-2)
    sampled_pred_tensor = torch.cat(sampled_pred_tensor, dim=-2)

    return chamfer_loss_vectorized(sampled_gt_tensor, sampled_pred_tensor)


def compute_iou(svg_pred, svg_gt, size=64, thr=255 * 3 / 4):
    """
    Compute pixel-level Intersection over Union (IoU) between two SVGs.

    Args:
        svg_pred: Predicted SVG object.
        svg_gt: Ground-truth SVG object.
        size (int): Rasterization size.
        thr (float): Threshold for foreground binarization.

    Returns:
        float: IoU score.
    """
    svg_pred = svg_pred.copy().tighten_viewbox(pad=0.5)
    svg_gt = svg_gt.copy().tighten_viewbox(pad=0.5)

    pred_L = svg_to_png(svg_pred, size=size)
    pred_arr = np.array(pred_L)

    gt_L = svg_to_png(svg_gt, size=size)
    gt_arr = np.array(gt_L)

    mask_pred = pred_arr < thr
    mask_gt = gt_arr < thr

    inter = np.sum(mask_pred & mask_gt)
    union = np.sum(mask_pred | mask_gt)

    iou = (inter / union) if union > 0 else 1.0
    return iou


def compute_l1(svg_pred, svg_gt, size=64, thr=255 * 3 / 4):
    """
    Compute pixel-wise L1 distance between binary masks of two SVGs.

    Args:
        svg_pred: Predicted SVG object.
        svg_gt: Ground-truth SVG object.
        size (int): Rasterization size.
        thr (float): Threshold for foreground binarization.

    Returns:
        float: Mean absolute mask difference.
    """
    svg_pred = svg_pred.copy().tighten_viewbox(pad=0.05)
    svg_gt = svg_gt.copy().tighten_viewbox(pad=0.05)

    pred_L = svg_to_png(svg_pred, size=size)
    gt_L = svg_to_png(svg_gt, size=size)

    pred_arr = np.array(pred_L)
    gt_arr = np.array(gt_L)

    mask_pred = pred_arr < thr
    mask_gt = gt_arr < thr

    dist = np.mean(np.abs(mask_pred.astype(float) - mask_gt.astype(float)))
    return dist


def _as_six_arg_tensor(args: torch.Tensor) -> torch.Tensor:
    if args.size(-1) == 6:
        return args
    if args.size(-1) == 8:
        return args[..., -6:]
    raise ValueError(f"Expected 6- or 8-argument SVG tensor, got last dimension {args.size(-1)}")


def _prepend_sos(commands: torch.Tensor, args: torch.Tensor, sos_id: int = 4, pad_val: float = -1.0):
    sos_cmds = commands.new_full((*commands.shape[:-1], 1), sos_id)
    sos_args = args.new_full((*args.shape[:-2], 1, args.size(-1)), pad_val)
    return torch.cat([sos_cmds, commands], dim=-1), torch.cat([sos_args, args], dim=-2)


def geometric_constraint_correctness(
    gt_cmds: torch.Tensor,
    gt_args: torch.Tensor,
    pred_cmds: torch.Tensor,
    pred_args: torch.Tensor,
    *,
    constraint: str,
    gt_labels: torch.Tensor | None = None,
) -> list[float]:
    """
    Compare geometry-derived continuity/alignment labels for decoded SVG tensors.

    The ground-truth tensors are expected to include SOS. Predicted tensors are
    expected to come from greedy decoding without SOS, so SOS is prepended before
    predicted labels are computed. If ``gt_labels`` is provided, it is used
    directly; otherwise ground-truth labels are recomputed from geometry.
    Only valid ground-truth constraint positions are scored.
    """
    if constraint == "continuity":
        label_fn = compute_continuity_tensor
    elif constraint == "alignment":
        label_fn = compute_line_alignment_tensor
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    gt_args = _as_six_arg_tensor(gt_args)
    pred_args = _as_six_arg_tensor(pred_args)

    if gt_cmds.ndim != 3 or pred_cmds.ndim != 3:
        raise ValueError(
            "Expected command tensors shaped (N,G,S). " f"Got gt={tuple(gt_cmds.shape)}, pred={tuple(pred_cmds.shape)}"
        )

    if gt_labels is not None and gt_labels.ndim != 3:
        raise ValueError(f"Expected gt_labels shaped (N,G,S), got {tuple(gt_labels.shape)}")

    if gt_args.ndim != 4 or pred_args.ndim != 4:
        raise ValueError(
            "Expected argument tensors shaped (N,G,S,D). "
            f"Got gt={tuple(gt_args.shape)}, pred={tuple(pred_args.shape)}"
        )

    if gt_cmds.shape[:2] != pred_cmds.shape[:2] or gt_args.shape[:2] != pred_args.shape[:2]:
        raise ValueError(
            "Ground-truth and predicted tensors must have matching sample/group dimensions. "
            f"Got gt_cmds={tuple(gt_cmds.shape)}, pred_cmds={tuple(pred_cmds.shape)}, "
            f"gt_args={tuple(gt_args.shape)}, pred_args={tuple(pred_args.shape)}"
        )

    if gt_labels is not None and gt_labels.shape[:2] != pred_cmds.shape[:2]:
        raise ValueError(
            "Ground-truth labels and predicted tensors must have matching sample/group dimensions. "
            f"Got gt_labels={tuple(gt_labels.shape)}, pred_cmds={tuple(pred_cmds.shape)}"
        )

    pred_cmds, pred_args = _prepend_sos(pred_cmds, pred_args)

    correctness: list[float] = []
    n_samples, n_groups = gt_cmds.shape[:2]

    for sample_idx in range(n_samples):
        for group_idx in range(n_groups):
            if gt_labels is None:
                gt_labels_i = label_fn(gt_cmds[sample_idx, group_idx], gt_args[sample_idx, group_idx])
            else:
                gt_labels_i = gt_labels[sample_idx, group_idx]
            pred_labels = label_fn(pred_cmds[sample_idx, group_idx], pred_args[sample_idx, group_idx])

            seq_len = min(gt_labels_i.size(0), pred_labels.size(0))
            valid = gt_labels_i[:seq_len] != -1
            if valid.any():
                correctness.extend(pred_labels[:seq_len][valid].eq(gt_labels_i[:seq_len][valid]).float().cpu().tolist())

    return correctness

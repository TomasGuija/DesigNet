from __future__ import annotations

import torch

RTOL = 0.05
ATOL = 0.01


def compute_line_alignment_tensor(cmds: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
    """Classify line commands as horizontal, vertical, or other."""
    n = cmds.shape[0]
    alignment = torch.full((n,), -1, dtype=torch.long, device=cmds.device)

    for i in range(1, n):
        if cmds[i] != 1:
            continue

        x1, y1 = args[i - 1][-2:].tolist()
        x2, y2 = args[i][-2:].tolist()
        dx, dy = x2 - x1, y2 - y1

        if dx == 0 and dy == 0:
            alignment[i] = 2
        elif dy == 0:
            alignment[i] = 0
        elif dx == 0:
            alignment[i] = 1
        else:
            alignment[i] = 2

    return alignment


def compute_continuity_tensor(cmds: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
    """Compute endpoint continuity class for each command."""

    def compute_angle(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        v1_norm = v1 / v1.norm().clamp(min=eps)
        v2_norm = v2 / v2.norm().clamp(min=eps)
        dot_product = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
        return torch.acos(dot_product)

    def check_continuity(prev_dir: torch.Tensor, new_dir: torch.Tensor, eps: float = 5e-2) -> int:
        if prev_dir.norm() < 1e-6 or new_dir.norm() < 1e-6:
            return 0

        length_match = torch.isclose(prev_dir.norm(), new_dir.norm(), rtol=RTOL, atol=ATOL)
        angle = compute_angle(prev_dir, new_dir)

        if angle < eps and length_match:
            return 2
        if angle < eps:
            return 1
        return 0

    n = cmds.shape[0]
    continuity = torch.full((n,), -1, dtype=torch.long, device=cmds.device)

    last_endpoint = args[1][-2:]

    for i in range(2, n - 1):
        if cmds[i] == 3:
            break

        current_endpoint = args[i][-2:]

        if cmds[i + 1] == 3:
            next_command = cmds[2]
            next_args = args[2]
        else:
            next_command = cmds[i + 1]
            next_args = args[i + 1]

        if cmds[i] == 1:
            continuity[i] = -1
            if next_command == 2:
                ctrl1 = next_args[:2]
                prev_dir = current_endpoint - last_endpoint
                new_dir = ctrl1 - current_endpoint
                continuity[i] = check_continuity(prev_dir, new_dir)

        elif cmds[i] == 2:
            continuity[i] = 0
            ctrl2 = args[i][2:-2]
            prev_dir = current_endpoint - ctrl2

            if next_command == 1:
                next_endpoint = next_args[-2:]
                new_dir = next_endpoint - current_endpoint
                continuity[i] = check_continuity(prev_dir, new_dir)
            elif next_command == 2:
                ctrl1 = next_args[:2]
                new_dir = ctrl1 - current_endpoint
                continuity[i] = check_continuity(prev_dir, new_dir)

        last_endpoint = current_endpoint

    return continuity

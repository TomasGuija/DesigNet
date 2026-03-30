import torch

from designet.difflib.tensor import SVGTensor


def get_padding_mask(commands, seq_dim=1, extended=False):
    padding_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).cumsum(dim=seq_dim) == 0
    padding_mask = padding_mask.float()

    if extended:
        # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
        S = commands.size(seq_dim)
        torch.narrow(padding_mask, seq_dim, 3, S - 3).add_(torch.narrow(padding_mask, seq_dim, 0, S - 3)).clamp_(max=1)

    if seq_dim == 0:
        return padding_mask.unsqueeze(-1)
    return padding_mask


def canonicalize_endpoints_8args(cmds, args):
    """
    Canonicalize endpoints in 8-arg SVG commands so that each on-contour joint
    has a single averaged position.

    Args:
        cmds:  (B, G, S) LongTensor — command ids.
        args:  (B, G, S, 8) FloatTensor — start, ctrl1, ctrl2, end coords.

    Returns:
        canon_args: (B, G, S, 8) FloatTensor — same as args but with
                    endpoints canonicalized.
    """
    assert args.size(-1) == 8, "Expected 8-arg commands"

    args = args.clone()

    B, G, S, _ = args.shape
    device = args.device

    start = args[..., :2]
    end = args[..., -2:]

    # roll-based neighbour starts/ends
    prev_end = end.roll(shifts=1, dims=2)
    next_start = start.roll(shifts=-1, dims=2)

    # Identify wrap joint indices
    eos_pos = torch.argmax((cmds == 3).to(torch.int64), dim=2)  # (B,G)
    last_cmd = eos_pos - 1  # (B,G)
    first_draw = torch.ones_like(last_cmd)  # == 1

    # Index helpers
    bidx = torch.arange(B, device=device)[:, None, None]
    gidx = torch.arange(G, device=device)[None, :, None]
    last_idx = last_cmd.unsqueeze(-1)  # (B,G,1)
    first_idx = first_draw.unsqueeze(-1)  # (B,G,1)

    # For prev_end of first_draw, use end[last_cmd]
    prev_end[bidx, gidx, first_idx] = end[bidx, gidx, last_idx]
    # For prev_end of MOVE (index 0), also use end[last_cmd]
    prev_end[bidx, gidx, 0] = end[bidx, gidx, last_idx]
    # For next_start of last_cmd, use start[first_draw]
    next_start[bidx, gidx, last_idx] = start[bidx, gidx, first_idx]

    # Base canonical start/end
    canonical_start = 0.5 * (prev_end + start)
    canonical_end = 0.5 * (end + next_start)

    # Wrap joint 3-way average: MOVE end, first_draw start, last_cmd end
    move_end = end[:, :, 0:1, :]  # (B,G,1,2)
    last_end = end[bidx, gidx, last_idx]  # (B,G,1,2)
    first_start = start[bidx, gidx, first_idx]  # (B,G,1,2)
    wrap_joint3 = (last_end + first_start + move_end) / 3.0

    # Override canonical wrap joints
    canonical_start[bidx, gidx, first_idx] = wrap_joint3
    canonical_end[bidx, gidx, 0] = wrap_joint3
    canonical_end[bidx, gidx, last_idx] = wrap_joint3

    # Assign canonical endpoints back into args
    args[..., -2:] = canonical_end
    args[..., :2] = canonical_start

    return args

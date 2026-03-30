import torch
import torch.nn as nn
import torch.nn.functional as F

from designet.svglib.utils import canonicalize_endpoints_8args


def st_argmax(logits: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable argmax via straight-through estimator.
    Forward: hard one-hot argmax.
    Backward: gradients as if softmax(logits / tau) was used.

    Args:
        logits: (..., K) unnormalized scores
        dim: dimension of classes
        tau: temperature for softmax in backward (smaller -> sharper)
    Returns:
        y: one-hot-like tensor with same shape as logits
    """
    # soft distribution used for gradients
    y_soft = F.softmax(logits / tau, dim=dim)

    # hard argmax one-hot used for forward
    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)

    # straight-through: forward uses y_hard, backward uses y_soft
    y = y_hard.detach() - y_soft.detach() + y_soft
    return y


class SoftAlignmentRefinerBatched(nn.Module):
    """
    Differentiable axis-alignment refiner for 8-argument command format.

    This module applies the predicted alignment class (horizontal, vertical,
    or none) to line segments while keeping the operation differentiable via a
    straight-through estimator. It also propagates endpoint updates to the
    following segment start and handles the wrap-around junction of closed
    contours.

    Expected:
      - args8 shape: (B,G,S,8)
      - END is args[..., -2:]
      - START is args[..., :2]
      - align_logits over 3 classes, e.g. [H, V, NONE] or any ordering you set via indices.
    """

    def __init__(self, tau: float = 1.0, MOVE_ID=0, EOS_ID=3, h_idx=0, v_idx=1, none_idx=2):
        super().__init__()
        self.tau = tau
        self.MOVE_ID = MOVE_ID
        self.EOS_ID = EOS_ID
        self.h_idx = h_idx
        self.v_idx = v_idx
        self.none_idx = none_idx

    def _last_idx(self, cmds):
        eos = cmds == self.EOS_ID
        has = eos.any(dim=2)
        first_eos = eos.int().argmax(dim=2)
        L = torch.where(has, first_eos - 1, torch.full_like(first_eos, cmds.size(2) - 1))
        return L.clamp(min=0, max=cmds.size(2) - 1)

    def forward(self, cmds, args8, align_logits):
        """
        Refine line alignment from predicted alignment logits.

        Args:
            cmds: (B, G, S) command ids.
            args8: (B, G, S, 8) command arguments with explicit start/end.
            align_logits: (B, G, S, 3) logits for [H, V, NONE] (or mapped indices).

        Returns:
            Refined 8-argument tensor with snapped line endpoints.
        """

        B, G, S, D = args8.shape
        assert D == 8, "Expected 8-arg commands"

        refined = args8.clone()

        # canonicalize (important in your pipeline)
        canonical = canonicalize_endpoints_8args(cmds, refined)
        start = canonical[..., :2]  # (B,G,S,2)
        end = canonical[..., -2:]  # (B,G,S,2)

        # probabilities
        p = st_argmax(align_logits, dim=-1, tau=self.tau)  # (B,G,S,3)

        p_h = p[..., self.h_idx : self.h_idx + 1]
        p_v = p[..., self.v_idx : self.v_idx + 1]
        p_none = p[..., self.none_idx : self.none_idx + 1]

        is_line = (cmds == 1).unsqueeze(-1)  # (B,G,S,1)

        # candidate ENDs
        end_none = end
        end_h = torch.stack([end[..., 0], start[..., 1]], dim=-1)  # END.y := START.y
        end_v = torch.stack([start[..., 0], end[..., 1]], dim=-1)  # END.x := START.x

        end_soft = p_none * end_none + p_h * end_h + p_v * end_v
        end_new = torch.where(is_line, end_soft, end)

        # --- Propagate END -> next START ---
        prop_values_start = end_new.roll(shifts=1, dims=2)
        prop_mask_start = is_line.roll(shifts=1, dims=2)  # next command's START affected by prev LINE
        start_new = torch.where(prop_mask_start, prop_values_start, args8[..., :2])

        # --- Wrap joint handling (last_cmd END -> MOVE END & first drawable START) ---
        L = self._last_idx(cmds)  # (B,G)
        b_idx = torch.arange(B, device=args8.device)[:, None].expand(B, G)
        g_idx = torch.arange(G, device=args8.device)[None, :].expand(B, G)

        last_end = end_new[b_idx, g_idx, L, :]  # (B,G,2)

        end_final = end_new.clone()
        start_final = start_new.clone()

        end_final[b_idx, g_idx, 0, :] = last_end
        start_final[b_idx, g_idx, 1, :] = last_end

        mid = args8[..., 2:-2]  # (B,G,S,4)

        refined = torch.cat([start_final, mid, end_final], dim=-1)

        return refined


class SoftContinuityRefinerBatched(nn.Module):
    """
    Differentiable continuity refiner for cubic/line junctions.

    Given continuity logits over {C0, G1, C1}, this module adjusts Bézier
    control points at segment junctions so predicted continuity constraints are
    better satisfied. It handles interior junctions and the closing junction of
    closed contours, and uses straight-through class selection for training.
    """

    def __init__(self, tau: float = 1.0, c0_idx=0, g1_idx=1, c1_idx=2, eps=1e-8):
        super().__init__()
        self.tau = tau
        self.c0_idx = c0_idx
        self.g1_idx = g1_idx
        self.c1_idx = c1_idx
        self.eps = eps

    def normalize(self, v):
        """Normalize vectors with epsilon stabilization."""
        return v / (v.norm(dim=-1, keepdim=True) + self.eps)

    def enforce_g1_cp2(self, endpoint, ctrl2, target_dir):
        """Project incoming tangent direction for cp2 while keeping tangent length."""
        # incoming tangent is endpoint - ctrl2
        offset = endpoint - ctrl2
        length = offset.norm(dim=-1, keepdim=True)
        dir_norm = self.normalize(target_dir)
        return endpoint - dir_norm * length

    def enforce_g1_cp1(self, endpoint, ctrl1, target_dir):
        """Project outgoing tangent direction for cp1 while keeping tangent length."""
        # outgoing tangent is ctrl1 - endpoint
        offset = ctrl1 - endpoint
        length = offset.norm(dim=-1, keepdim=True)
        dir_norm = self.normalize(target_dir)
        return endpoint + dir_norm * length

    def _get_last_drawable_idx(self, cmds, eos_id=3):
        """Return last non-EOS index for each contour in the batch."""
        # last drawable index per (B,G): first EOS - 1, else S-1
        eos = cmds == eos_id
        has = eos.any(dim=2)
        first_eos = eos.int().argmax(dim=2)
        L = torch.where(has, first_eos - 1, torch.full_like(first_eos, cmds.size(2) - 1))
        return L.clamp(min=0, max=cmds.size(2) - 1)

    def forward(self, cmds, args6, cont_logits):
        """
        Refine Bézier control points from predicted continuity labels.

        Args:
            cmds: (B, G, S) command ids.
            args6: (B, G, S, 6) arguments in [c1x,c1y,c2x,c2y,ex,ey] format.
            cont_logits: (B, G, S, 3) logits for continuity classes {C0, G1, C1}.

        Returns:
            Refined args tensor with updated control points at affected junctions.
        """
        # probsç
        B, G, S, D = args6.shape

        p = st_argmax(cont_logits, dim=-1, tau=self.tau)  # (B,G,S,3)

        p_c0 = p[..., self.c0_idx : self.c0_idx + 1]
        p_g1 = p[..., self.g1_idx : self.g1_idx + 1]
        p_c1 = p[..., self.c1_idx : self.c1_idx + 1]

        endpoints = args6[..., -2:]
        ctrl1 = args6[..., :2]
        ctrl2 = args6[..., 2:4]

        ctrl1_new = ctrl1.clone()
        ctrl2_new = ctrl2.clone()

        # -------------------------
        # (A) Interior junctions
        # -------------------------
        curr_end = endpoints[:, :, 1:-1]
        prev_end = endpoints[:, :, :-2]
        next_end = endpoints[:, :, 2:]

        curr_ctrl2 = ctrl2[:, :, 1:-1]
        next_ctrl1 = ctrl1[:, :, 2:]

        curr_cmd = cmds[:, :, 1:-1]
        next_cmd = cmds[:, :, 2:]

        curr_line_dir = curr_end - prev_end
        next_line_dir = next_end - curr_end
        curr_curve_dir = curr_end - curr_ctrl2
        next_curve_dir = next_ctrl1 - curr_end

        # candidates
        # curve->line affects curr ctrl2
        cand_cp2_c0_c2l = curr_ctrl2
        cand_cp2_g1_c2l = self.enforce_g1_cp2(curr_end, curr_ctrl2, next_line_dir)
        cand_cp2_c1_c2l = curr_end - next_line_dir

        # line->curve affects next ctrl1
        cand_cp1_c0_l2c = next_ctrl1
        cand_cp1_g1_l2c = self.enforce_g1_cp1(curr_end, next_ctrl1, curr_line_dir)
        cand_cp1_c1_l2c = curr_end + curr_line_dir

        # curve->curve affects both
        avg_dir = self.normalize(self.normalize(curr_curve_dir) + self.normalize(next_curve_dir))
        len_in = curr_curve_dir.norm(dim=-1, keepdim=True)
        len_out = next_curve_dir.norm(dim=-1, keepdim=True)
        avg_len = 0.5 * (len_in + len_out)

        cand_cp2_c0_c2c = curr_ctrl2
        cand_cp1_c0_c2c = next_ctrl1

        cand_cp2_g1_c2c = curr_end - avg_dir * len_in
        cand_cp1_g1_c2c = curr_end + avg_dir * len_out

        cand_cp2_c1_c2c = curr_end - avg_dir * avg_len
        cand_cp1_c1_c2c = curr_end + avg_dir * avg_len

        # weights at interior junction indices (1:-1)
        w_c0 = p_c0[:, :, 1:-1]
        w_g1 = p_g1[:, :, 1:-1]
        w_c1 = p_c1[:, :, 1:-1]

        c2l = (curr_cmd == 2) & (next_cmd == 1)
        l2c = (curr_cmd == 1) & (next_cmd == 2)
        c2c = (curr_cmd == 2) & (next_cmd == 2)

        blended_cp2_c2l = w_c0 * cand_cp2_c0_c2l + w_g1 * cand_cp2_g1_c2l + w_c1 * cand_cp2_c1_c2l
        blended_cp1_l2c = w_c0 * cand_cp1_c0_l2c + w_g1 * cand_cp1_g1_l2c + w_c1 * cand_cp1_c1_l2c

        blended_cp2_c2c = w_c0 * cand_cp2_c0_c2c + w_g1 * cand_cp2_g1_c2c + w_c1 * cand_cp2_c1_c2c
        blended_cp1_c2c = w_c0 * cand_cp1_c0_c2c + w_g1 * cand_cp1_g1_c2c + w_c1 * cand_cp1_c1_c2c

        # Update ctrl2 for commands i=1..S-2 (index 1:-1)
        ctrl2_mid = ctrl2_new[:, :, 1:-1]
        ctrl2_mid = torch.where(c2l.unsqueeze(-1), blended_cp2_c2l, ctrl2_mid)
        ctrl2_mid = torch.where(c2c.unsqueeze(-1), blended_cp2_c2c, ctrl2_mid)
        ctrl2_new = torch.cat([ctrl2_new[:, :, :1], ctrl2_mid, ctrl2_new[:, :, -1:]], dim=2)

        # Update ctrl1 for commands i+1=2..S-1 (index 2:)
        ctrl1_tail = ctrl1_new[:, :, 2:]
        ctrl1_tail = torch.where(l2c.unsqueeze(-1), blended_cp1_l2c, ctrl1_tail)
        ctrl1_tail = torch.where(c2c.unsqueeze(-1), blended_cp1_c2c, ctrl1_tail)
        ctrl1_new = torch.cat([ctrl1_new[:, :, :2], ctrl1_tail], dim=2)

        # -------------------------
        # (B) Closing junction: last drawable -> first drawable
        # -------------------------
        L = self._get_last_drawable_idx(cmds, eos_id=3)  # (B,G)
        L_idx2 = L.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)

        last_end = endpoints.gather(2, L_idx2).squeeze(2)  # (B,G,2)
        last_cp2 = ctrl2_new.gather(2, L_idx2).squeeze(2)  # (B,G,2)
        last_cmd = cmds.gather(2, L.unsqueeze(-1)).squeeze(-1)  # (B,G)

        prevL = (L - 1).clamp(min=0)
        prev_end = endpoints.gather(2, prevL.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
        last_line_dir = last_end - prev_end  # (B,G,2)

        # first drawable is index 1 in your convention
        first_cmd = cmds[:, :, 1]
        first_end = endpoints[:, :, 1]
        first_cp1 = ctrl1_new[:, :, 1]

        first_line_dir = first_end - last_end
        first_curve_dir = first_cp1 - last_end
        last_curve_dir = last_end - last_cp2

        # weights from cont_logits at the closing junction index L
        w_close = p.gather(2, L.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, p.size(-1))).squeeze(2)  # (B,G,3)
        w0 = w_close[:, :, self.c0_idx : self.c0_idx + 1]
        w1 = w_close[:, :, self.g1_idx : self.g1_idx + 1]
        w2 = w_close[:, :, self.c1_idx : self.c1_idx + 1]

        # structure cases for closing
        close_c2l = (last_cmd == 2) & (first_cmd == 1)
        close_l2c = (last_cmd == 1) & (first_cmd == 2)
        close_c2c = (last_cmd == 2) & (first_cmd == 2)

        # candidates for closing:
        # last curve -> first line : update last cp2
        cand_lastcp2_c0 = last_cp2
        cand_lastcp2_g1 = self.enforce_g1_cp2(last_end, last_cp2, first_line_dir)
        cand_lastcp2_c1 = last_end - first_line_dir

        blended_lastcp2 = w0 * cand_lastcp2_c0 + w1 * cand_lastcp2_g1 + w2 * cand_lastcp2_c1

        # last line -> first curve : update first cp1
        cand_firstcp1_c0 = first_cp1
        cand_firstcp1_g1 = self.enforce_g1_cp1(last_end, first_cp1, last_line_dir)
        cand_firstcp1_c1 = last_end + last_line_dir

        blended_firstcp1_l2c = w0 * cand_firstcp1_c0 + w1 * cand_firstcp1_g1 + w2 * cand_firstcp1_c1

        # curve -> curve closing: update both using avg_dir
        avg_dir_close = self.normalize(self.normalize(last_curve_dir) + self.normalize(first_curve_dir))
        len_last = last_curve_dir.norm(dim=-1, keepdim=True)
        len_first = first_curve_dir.norm(dim=-1, keepdim=True)
        avg_len_close = 0.5 * (len_last + len_first)

        cand_lastcp2_c2c_c0 = last_cp2
        cand_firstcp1_c2c_c0 = first_cp1

        cand_lastcp2_c2c_g1 = last_end - avg_dir_close * len_last
        cand_firstcp1_c2c_g1 = last_end + avg_dir_close * len_first

        cand_lastcp2_c2c_c1 = last_end - avg_dir_close * avg_len_close
        cand_firstcp1_c2c_c1 = last_end + avg_dir_close * avg_len_close

        blended_lastcp2_c2c = w0 * cand_lastcp2_c2c_c0 + w1 * cand_lastcp2_c2c_g1 + w2 * cand_lastcp2_c2c_c1
        blended_firstcp1_c2c = w0 * cand_firstcp1_c2c_c0 + w1 * cand_firstcp1_c2c_g1 + w2 * cand_firstcp1_c2c_c1

        # write back closing updates
        # last ctrl2 at index L
        # first ctrl1 at index 1
        # update last ctrl2 for curve->line and curve->curve
        new_last_cp2 = torch.where(close_c2l.unsqueeze(-1), blended_lastcp2, last_cp2)
        new_last_cp2 = torch.where(close_c2c.unsqueeze(-1), blended_lastcp2_c2c, new_last_cp2)

        new_first_cp1 = torch.where(close_l2c.unsqueeze(-1), blended_firstcp1_l2c, first_cp1)
        new_first_cp1 = torch.where(close_c2c.unsqueeze(-1), blended_firstcp1_c2c, new_first_cp1)

        # Write into ctrl2_new / ctrl1_new (these are the ONLY "writes", but into fresh clones)
        b_idx = torch.arange(B, device=args6.device)[:, None].expand(B, G)
        g_idx = torch.arange(G, device=args6.device)[None, :].expand(B, G)
        ctrl2_new = ctrl2_new.clone()
        ctrl1_new = ctrl1_new.clone()
        ctrl2_new[b_idx, g_idx, L, :] = new_last_cp2
        ctrl1_new[:, :, 1, :] = new_first_cp1

        # rebuild args6
        out = torch.cat([ctrl1_new, ctrl2_new, endpoints], dim=-1)
        return out

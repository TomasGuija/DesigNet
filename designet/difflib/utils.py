from designet.difflib.tensor import SVGTensor
from designet.svglib.geom import Bbox
from designet.svglib.svg import SVG
from designet.svglib.utils import get_padding_mask


def build_svg_from_pred_cmds(cmds, args, allow_empty=True):
    args = args[..., -6:]
    padding_mask = get_padding_mask(cmds).squeeze(-1).bool()  # (G,S)
    rows_to_clear = (padding_mask[:, 0]) & ~(padding_mask[:, 1:].any(dim=1))
    padding_mask[rows_to_clear, :] = False

    if not padding_mask.any():
        return None

    svg_tensor = SVGTensor.from_cmd_args(cmds[padding_mask].cpu(), args[padding_mask].cpu())

    if svg_tensor is None:
        return None

    vb = Bbox(-1.0, -1.0, 2, 2)

    return SVG.from_tensor(svg_tensor.data, viewbox=vb, allow_empty=allow_empty).split_paths()

from __future__ import annotations

from typing import List
from xml.dom import expatbuilder

import numpy as np
import torch

from .geom import Bbox, Point, union_bbox
from .svg_path import SVGPath
from .svg_primitive import SVGLine, SVGPathGroup


class SVG:
    def __init__(self, svg_path_groups: List[SVGPathGroup], viewbox: Bbox = None):
        self.svg_path_groups = svg_path_groups
        self.viewbox = Bbox(24) if viewbox is None else viewbox

    @staticmethod
    def load_svg(file_path):
        with open(file_path, "r") as f:
            return SVG.from_str(f.read())

    @staticmethod
    def from_str(svg_str: str):
        svg_path_groups = []
        svg_dom = expatbuilder.parseString(svg_str, False)
        svg_root = svg_dom.getElementsByTagName("svg")[0]

        viewbox_list = list(map(float, svg_root.getAttribute("viewBox").split(" ")))
        view_box = Bbox(*viewbox_list)

        primitives = {
            "path": SVGPath,
            "line": SVGLine,
        }
        for tag, primitive_cls in primitives.items():
            for x in svg_dom.getElementsByTagName(tag):
                svg_path_groups.append(primitive_cls.from_xml(x))

        return SVG(svg_path_groups, view_box)

    def _get_viz_elements(
        self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=False, with_moves=True
    ):
        viz_elements = []
        for svg_path_group in self.svg_path_groups:
            viz_elements.extend(
                svg_path_group._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves)
            )
        return viz_elements

    def _markers(self) -> str:
        return (
            "<defs>"
            '<marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="black" />'
            "</marker>"
            "</defs>"
        )

    def save_svg(self, file_path):
        with open(file_path, "w") as f:
            f.write(self.to_str())

    def to_str(
        self,
        fill=False,
        with_points=False,
        with_handles=False,
        with_bboxes=False,
        with_markers=False,
        color_firstlast=False,
        with_moves=True,
    ) -> str:
        viz_elements = self._get_viz_elements(
            with_points,
            with_handles,
            with_bboxes,
            color_firstlast,
            with_moves,
        )
        newline = "\n"

        height = self.viewbox.size.y
        transform = f'transform="scale(1,-1) translate(0, {-height})"'

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{self.viewbox.to_str()}" height="256px" width="256px">'
            f'{self._markers() if with_markers else ""}'
            f"<g {transform}>"
            f"{newline.join(svg_path_group.to_str(fill=fill, with_markers=with_markers) for svg_path_group in [*self.svg_path_groups, *viz_elements])}"
            "</g>"
            "</svg>"
        )

    def to_tensor(self, concat_groups=True, PAD_VAL=-1):
        group_tensors = [p.to_tensor(PAD_VAL=PAD_VAL) for p in self.svg_path_groups]
        if concat_groups:
            return torch.cat(group_tensors, dim=0)
        return group_tensors

    def to_fillings(self):
        return [p.path.filling for p in self.svg_path_groups]

    @staticmethod
    def from_tensor(tensor: torch.Tensor, viewbox: Bbox = None, allow_empty=False):
        if viewbox is None:
            viewbox = Bbox(24)
        return SVG([SVGPath.from_tensor(tensor, allow_empty=allow_empty)], viewbox=viewbox)

    def tighten_viewbox(self, pad: float = 0.0):
        content = self.bbox()
        if content is None:
            self.viewbox = Bbox(1.0, 1.0)
            return self

        padded_xy = content.xy - Point(pad, pad)
        padded_wh = content.size + Point(2 * pad, 2 * pad)
        self.translate(-padded_xy)
        self.viewbox = Bbox(padded_wh.x, padded_wh.y)
        return self

    def draw_matplotlib(
        self,
        ax=None,
        figsize=(8, 8),
        fill=False,
        show_points=True,
        show_handles=True,
        flip_y=False,
        padding=0.06,
    ):
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.path import Path

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)

        for path_group in self.svg_path_groups:
            for path in path_group.svg_paths:
                vertices, codes = path.to_matplotlib()
                if not vertices:
                    continue

                verts = np.asarray(vertices, dtype=float)
                codes_arr = np.asarray(codes, dtype=np.uint8)

                i = 0
                prev = None
                subpath_start = None
                while i < len(codes_arr):
                    code = codes_arr[i]
                    v = verts[i]

                    if code == Path.MOVETO:
                        prev = v
                        subpath_start = v
                        i += 1
                        continue

                    if code == Path.LINETO and prev is not None:
                        seg_verts = np.array([prev, v])
                        seg_codes = np.array([Path.MOVETO, Path.LINETO], dtype=np.uint8)
                        ax.add_patch(
                            patches.PathPatch(
                                Path(seg_verts, seg_codes), facecolor="none", edgecolor="#ffd166", lw=3, clip_on=False
                            )
                        )
                        if show_points:
                            ax.scatter(
                                seg_verts[:, 0],
                                seg_verts[:, 1],
                                marker="x",
                                color="#9E0059",
                                s=40,
                                zorder=10,
                                alpha=0.7,
                                clip_on=False,
                            )
                        prev = v
                        i += 1
                        continue

                    if code == Path.CURVE4 and prev is not None:
                        if i + 2 >= len(codes_arr) or not (
                            codes_arr[i] == codes_arr[i + 1] == codes_arr[i + 2] == Path.CURVE4
                        ):
                            prev = v
                            i += 1
                            continue

                        c1 = verts[i]
                        c2 = verts[i + 1]
                        p3 = verts[i + 2]

                        seg_verts = np.vstack([prev, c1, c2, p3])
                        seg_codes = np.array([Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4], dtype=np.uint8)
                        ax.add_patch(
                            patches.PathPatch(
                                Path(seg_verts, seg_codes),
                                facecolor=("black" if fill else "none"),
                                edgecolor="#06d6a0",
                                lw=3,
                                clip_on=False,
                            )
                        )

                        if show_handles:
                            ax.plot([prev[0], c1[0]], [prev[1], c1[1]], "--", color="#118ab2", alpha=0.5, lw=0.5)
                            ax.plot([c2[0], p3[0]], [c2[1], p3[1]], "--", color="#118ab2", alpha=0.5, lw=0.5)
                            ax.scatter([c1[0], c2[0]], [c1[1], c2[1]], marker="o", color="#118ab2", s=20, alpha=0.7)

                        if show_points:
                            ax.scatter(
                                [prev[0], p3[0]],
                                [prev[1], p3[1]],
                                marker="x",
                                color="#9E0059",
                                s=40,
                                zorder=10,
                                alpha=0.7,
                            )

                        prev = p3
                        i += 3
                        continue

                    if code == Path.CLOSEPOLY and prev is not None and subpath_start is not None:
                        seg_verts = np.array([prev, subpath_start])
                        seg_codes = np.array([Path.MOVETO, Path.LINETO], dtype=np.uint8)
                        ax.add_patch(
                            patches.PathPatch(
                                Path(seg_verts, seg_codes), facecolor="none", edgecolor="#118ab2", lw=3, clip_on=False
                            )
                        )
                        if show_points:
                            ax.scatter(
                                seg_verts[:, 0],
                                seg_verts[:, 1],
                                marker="x",
                                color="#9E0059",
                                s=40,
                                zorder=10,
                                alpha=0.7,
                            )

                        prev = subpath_start
                        i += 1
                        continue

                    prev = v
                    i += 1

    def draw(self, *args, **kwargs):
        return self.draw_matplotlib(*args, **kwargs)

    def _apply_to_paths(self, method, *args, **kwargs):
        for path_group in self.svg_path_groups:
            getattr(path_group, method)(*args, **kwargs)
        return self

    def split_paths(self):
        path_groups = []
        for path_group in self.svg_path_groups:
            path_groups.extend(path_group.split_paths())
        self.svg_path_groups = path_groups
        return self

    def empty(self):
        return len(self.svg_path_groups) == 0

    def translate(self, vec: Point):
        return self._apply_to_paths("translate", vec)

    def bbox(self):
        return union_bbox([path_group.bbox() for path_group in self.svg_path_groups])

    def to_points(self, sort=True):
        if not self.svg_path_groups:
            return np.empty((0, 2))

        points = np.concatenate([path_group.to_points() for path_group in self.svg_path_groups])
        if points.shape[0] == 0:
            return np.empty((0, 2))

        if sort:
            ind = np.lexsort((points[:, 0], points[:, 1]))
            points = points[ind]
            row_mask = np.append([True], np.any(np.diff(points, axis=0), 1))
            points = points[row_mask]
        else:
            _, idx = np.unique(points, axis=0, return_index=True)
            idx.sort()
            points = points[idx]

        return points

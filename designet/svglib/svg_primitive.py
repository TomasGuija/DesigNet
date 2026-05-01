from __future__ import annotations

from typing import List
from xml.dom import minidom

import numpy as np
import torch

from .geom import Point, Radius, union_bbox
from .svg_command import SVGCommandLine
from .svg_path import SVGPath


class SVGPrimitive:
    def __init__(self, color="black", fill=False, dasharray=None, stroke_width=".005", opacity=1.0):
        self.color = color
        self.dasharray = dasharray
        self.stroke_width = stroke_width
        self.opacity = opacity
        self.fill = fill

    def _get_fill_attr(self):
        fill_attr = (
            f'fill="{self.color}" fill-opacity="{self.opacity}"'
            if self.fill
            else f'fill="none" stroke="{self.color}" stroke-width="{self.stroke_width}" stroke-opacity="{self.opacity}"'
        )
        if self.dasharray is not None and not self.fill:
            fill_attr += f' stroke-dasharray="{self.dasharray}"'
        return fill_attr

    @classmethod
    def from_xml(cls, x: minidom.Element):
        raise NotImplementedError


class SVGLine(SVGPrimitive):
    def __init__(self, start_pos: Point, end_pos: Point, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_pos = start_pos
        self.end_pos = end_pos

    def __repr__(self):
        return f"SVGLine(xy1={self.start_pos} xy2={self.end_pos})"

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<line {fill_attr} x1="{self.start_pos.x}" y1="{self.start_pos.y}" x2="{self.end_pos.x}" y2="{self.end_pos.y}"/>'

    @classmethod
    def from_xml(cls, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"
        start_pos = Point(float(x.getAttribute("x1") or 0.0), float(x.getAttribute("y1") or 0.0))
        end_pos = Point(float(x.getAttribute("x2") or 0.0), float(x.getAttribute("y2") or 0.0))
        return cls(start_pos, end_pos, fill=fill)

    def to_path(self):
        return SVGPath([SVGCommandLine(self.start_pos, self.end_pos)]).to_group(fill=self.fill)


class SVGPathGroup(SVGPrimitive):
    def __init__(self, svg_paths: List[SVGPath] = None, origin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svg_paths = [] if svg_paths is None else svg_paths
        self.origin = Point(0.0) if origin is None else origin

    @property
    def paths(self):
        return self.svg_paths

    @property
    def path(self):
        return self.svg_paths[0]

    def __getitem__(self, idx):
        return self.svg_paths[idx]

    def __len__(self):
        return len(self.paths)

    def total_len(self):
        return sum(len(path) for path in self.svg_paths)

    @property
    def start_pos(self):
        return self.svg_paths[0].start_pos

    @property
    def end_pos(self):
        last_path = self.svg_paths[-1]
        return last_path.start_pos if last_path.closed else last_path.end_pos

    def copy(self):
        return SVGPathGroup(
            [svg_path.copy() for svg_path in self.svg_paths],
            self.origin.copy(),
            self.color,
            self.fill,
            self.dasharray,
            self.stroke_width,
            self.opacity,
        )

    def __repr__(self):
        return "SVGPathGroup({})".format(", ".join(svg_path.__repr__() for svg_path in self.svg_paths))

    def _get_viz_elements(
        self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True
    ):
        viz_elements = []
        for svg_path in self.svg_paths:
            viz_elements.extend(
                svg_path._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves)
            )
        return viz_elements

    def to_str(self, with_markers=False, *args, **kwargs):
        fill = kwargs.get("fill", self.fill)
        self.fill = fill
        fill_attr = self._get_fill_attr()
        marker_attr = 'marker-start="url(#arrow)"' if with_markers else ""
        d_attr = " ".join(svg_path.to_str() for svg_path in self.svg_paths)
        return f'<path {fill_attr} {marker_attr} filling="{self.path.filling}" d="{d_attr}"></path>'

    def numericalize(self, n=256, round_coords=True):
        return self._apply_to_paths("numericalize", n, round_coords)

    def to_str_commands_only(self):
        return " ".join(svg_path.to_str() for svg_path in self.svg_paths)

    def to_tensor(self, PAD_VAL=-1):
        return torch.cat([p.to_tensor(PAD_VAL=PAD_VAL) for p in self.svg_paths], dim=0)

    def _apply_to_paths(self, method, *args, **kwargs):
        for path in self.svg_paths:
            getattr(path, method)(*args, **kwargs)
        return self

    def translate(self, vec):
        return self._apply_to_paths("translate", vec)

    def scale(self, factor):
        return self._apply_to_paths("scale", factor)

    def split_paths(self):
        return [
            SVGPathGroup(
                [svg_path], self.origin, self.color, self.fill, self.dasharray, self.stroke_width, self.opacity
            )
            for svg_path in self.svg_paths
        ]

    def bbox(self):
        return union_bbox([path.bbox() for path in self.svg_paths])

    def to_points(self):
        if not self.svg_paths:
            return np.empty((0, 2))
        points = [path.to_points() for path in self.svg_paths]
        points = [p for p in points if p.shape[0] > 0]
        if not points:
            return np.empty((0, 2))
        return np.concatenate(points)


class SVGEllipse(SVGPrimitive):
    def __init__(self, center: Point, radius: Radius, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"SVGEllipse(c={self.center} r={self.radius})"

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<ellipse {fill_attr} cx="{self.center.x}" cy="{self.center.y}" rx="{self.radius.x}" ry="{self.radius.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("rx")), float(x.getAttribute("ry")))
        return SVGEllipse(center, radius, fill=fill)


class SVGCircle(SVGEllipse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"SVGCircle(c={self.center} r={self.radius})"

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<circle {fill_attr} cx="{self.center.x}" cy="{self.center.y}" r="{self.radius.x}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("r")))
        return SVGCircle(center, radius, fill=fill)

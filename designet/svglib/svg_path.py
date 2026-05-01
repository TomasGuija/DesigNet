from __future__ import annotations

import math
import re
from typing import List, Tuple
from xml.dom import minidom

import numpy as np
import torch

from ..difflib.tensor import SVGTensor
from .geom import Bbox, Point, union_bbox
from .svg_command import SVGCommand, SVGCommandClose, SVGCommandMove

COMMANDS = "MmZzLlHhVvCcSsQqTtAa"
COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

EMPTY_COMMAND = SVGCommandMove(Point(0.0))


class Filling:
    OUTLINE = 0
    FILL = 1
    ERASE = 2


class SVGPath:
    def __init__(
        self, path_commands: List[SVGCommand] = None, origin: Point = None, closed=False, filling=Filling.OUTLINE
    ):
        self.origin = origin or Point(0.0)
        self.path_commands = [] if path_commands is None else path_commands
        self.closed = closed
        self.filling = filling

    @property
    def start_command(self):
        return SVGCommandMove(self.origin, self.start_pos)

    @property
    def start_pos(self):
        return self.path_commands[0].start_pos

    @property
    def end_pos(self):
        return self.path_commands[-1].end_pos

    def to_group(self, *args, **kwargs):
        from .svg_primitive import SVGPathGroup

        return SVGPathGroup([self], *args, **kwargs)

    def __len__(self):
        return 1 + len(self.path_commands)

    def __getitem__(self, idx):
        if idx == 0:
            return self.start_command
        return self.path_commands[idx - 1]

    def all_commands(self, with_close=True):
        close_cmd = (
            [SVGCommandClose(self.path_commands[-1].end_pos.copy(), self.start_pos.copy())]
            if self.closed and self.path_commands and with_close
            else ()
        )
        return [self.start_command, *self.path_commands, *close_cmd]

    def copy(self):
        return SVGPath(
            [path_command.copy() for path_command in self.path_commands],
            self.origin.copy(),
            self.closed,
            filling=self.filling,
        )

    @staticmethod
    def _tokenize_path(path_str):
        cmd = None
        for x in COMMAND_RE.split(path_str):
            if x and x in COMMANDS:
                cmd = x
            elif cmd is not None:
                yield cmd, list(map(float, FLOAT_RE.findall(x)))

    @staticmethod
    def from_xml(x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"
        filling = Filling.OUTLINE if not x.hasAttribute("filling") else int(x.getAttribute("filling"))

        s = x.getAttribute("d")
        return SVGPath.from_str(s, fill=fill, filling=filling)

    @staticmethod
    def from_str(s: str, fill=False, filling=Filling.OUTLINE, add_closing=False):
        path_commands = []
        pos = initial_pos = Point(0.0)
        prev_command = None
        for cmd, args in SVGPath._tokenize_path(s):
            cmd_parsed, pos, initial_pos = SVGCommand.from_str(cmd, args, pos, initial_pos, prev_command)
            prev_command = cmd_parsed[-1]
            path_commands.extend(cmd_parsed)

        return SVGPath.from_commands(path_commands, fill=fill, filling=filling, add_closing=add_closing)

    @staticmethod
    def from_tensor(tensor: torch.Tensor, allow_empty=False):
        tensor = tensor[tensor[:, 0] != SVGTensor.COMMANDS_SIMPLIFIED.index("SOS")]
        tensor = tensor[tensor[:, 0] != SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")]
        return SVGPath.from_commands([SVGCommand.from_tensor(row) for row in tensor], allow_empty=allow_empty)

    @staticmethod
    def from_commands(
        path_commands: List[SVGCommand], fill=False, filling=Filling.OUTLINE, add_closing=False, allow_empty=False
    ):
        from .svg_command import SVGCommandMove
        from .svg_primitive import SVGPathGroup

        if not path_commands:
            return SVGPathGroup([])

        svg_paths = []
        svg_path = None

        for command in path_commands:
            if isinstance(command, SVGCommandMove):
                if svg_path is not None and (allow_empty or svg_path.path_commands):
                    if add_closing:
                        svg_path.closed = True
                    if not svg_path.path_commands:
                        svg_path.path_commands.append(EMPTY_COMMAND)
                    svg_paths.append(svg_path)

                svg_path = SVGPath([], command.start_pos.copy(), filling=filling)
            else:
                if svg_path is None:
                    continue

                if isinstance(command, SVGCommandClose):
                    if allow_empty or svg_path.path_commands:
                        svg_path.closed = True
                        if not svg_path.path_commands:
                            svg_path.path_commands.append(EMPTY_COMMAND)
                        svg_paths.append(svg_path)
                    svg_path = None
                else:
                    svg_path.path_commands.append(command)

        if svg_path is not None and (allow_empty or svg_path.path_commands):
            if add_closing:
                svg_path.closed = True
            if not svg_path.path_commands:
                svg_path.path_commands.append(EMPTY_COMMAND)
            svg_paths.append(svg_path)

        return SVGPathGroup(svg_paths, fill=fill)

    def to_str(self, fill=False):
        return " ".join(command.to_str() for command in self.all_commands())

    def _get_viz_elements(
        self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=False, with_moves=True
    ):
        points = self._get_points_viz(color_firstlast, with_moves) if with_points else ()
        handles = self._get_handles_viz() if with_handles else ()
        return [*points, *handles]

    def draw(self, viewbox=Bbox(24), *args, **kwargs):
        from .svg import SVG

        return SVG([self.to_group()], viewbox=viewbox).draw(*args, **kwargs)

    def _get_points_viz(self, color_firstlast=True, with_moves=True):
        points = []
        commands = self.all_commands(with_close=False)
        n = len(commands)
        for i, command in enumerate(commands):
            if not isinstance(command, SVGCommandMove) or with_moves:
                points_viz = command.get_points_viz(
                    first=(color_firstlast and i <= 1), last=(color_firstlast and i >= n - 2)
                )
                points.extend(points_viz)
        return points

    def _get_handles_viz(self):
        handles = []
        for command in self.path_commands:
            handles.extend(command.get_handles_viz())
        return handles

    def to_tensor(self, PAD_VAL=-1):
        return torch.stack([command.to_tensor(PAD_VAL=PAD_VAL) for command in self.all_commands()])

    def _get_unique_geoms(self):
        geoms = []
        for command in self.all_commands():
            geoms.extend(command.get_geoms())
        return list(set(geoms))

    def translate(self, vec):
        for i_geom in self._get_unique_geoms():
            i_geom.translate(vec)
        return self

    def scale(self, factor):
        for i_geom in self._get_unique_geoms():
            i_geom.scale(factor)
        return self

    def numericalize(self, n=256, round_coords=True):
        for command in self.all_commands():
            command.numericalize(n, round_coords)

    def bbox(self):
        return union_bbox([cmd.bbox() for cmd in self.path_commands])

    def to_points(self):
        return np.array([self.start_pos.pos, *(cmd.end_pos.pos for cmd in self.path_commands)])

    def to_matplotlib(self) -> Tuple[List[List[float]], List[int]]:
        MOVETO = 1
        LINETO = 2
        CURVE4 = 4
        CLOSEPOLY = 79

        vertices, codes = [], []

        if self.path_commands:
            vertices.append([self.path_commands[0].start_pos.x, self.path_commands[0].start_pos.y])
            codes.append(MOVETO)

        for cmd in self.path_commands:
            if cmd.command.name == "MOVE_TO":
                vertices.append([cmd.end_pos.x, cmd.end_pos.y])
                codes.append(MOVETO)
            elif cmd.command.name == "LINE_TO":
                vertices.append([cmd.end_pos.x, cmd.end_pos.y])
                codes.append(LINETO)
            elif cmd.command.name == "CUBIC_BEZIER":
                vertices.extend(
                    [[cmd.control1.x, cmd.control1.y], [cmd.control2.x, cmd.control2.y], [cmd.end_pos.x, cmd.end_pos.y]]
                )
                codes.extend([CURVE4, CURVE4, CURVE4])

        if self.closed and vertices:
            vertices.append([0, 0])
            codes.append(CLOSEPOLY)

        return vertices, codes

    def sample_points(self, max_dist=0.4):
        points = []

        for command in self.path_commands:
            try:
                length = command.length()
                n = max(math.ceil(length / max_dist), 1)
                sampled = command.sample_points(n=n, return_array=True)
                if sampled is not None and len(sampled) > 0:
                    points.extend(sampled)
            except (NotImplementedError, AttributeError, ValueError, TypeError):
                # Skip command if length or sampling fails
                continue
        return points

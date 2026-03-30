from __future__ import annotations

from enum import Enum
from typing import List, Union

import numpy as np
import torch

from designet.difflib.tensor import SVGTensor
from designet.svglib.geom import Bbox, Geom, Point, Radius
from designet.svglib.util_fns import get_roots

Num = Union[int, float]


class SVGCmdEnum(Enum):
    MOVE_TO = "m"
    LINE_TO = "l"
    CUBIC_BEZIER = "c"
    CLOSE_PATH = "z"


svgCmdArgTypes = {
    SVGCmdEnum.MOVE_TO.value: [Point],
    SVGCmdEnum.LINE_TO.value: [Point],
    SVGCmdEnum.CUBIC_BEZIER.value: [Point, Point, Point],
    SVGCmdEnum.CLOSE_PATH.value: [],
}


class SVGCommand:
    def __init__(self, command: SVGCmdEnum, args: List[Geom], start_pos: Point, end_pos: Point):
        self.command = command
        self.args = args

        self.start_pos = start_pos
        self.end_pos = end_pos

    def copy(self):
        raise NotImplementedError

    @staticmethod
    def from_str(cmd_str: str, args_str: List[Num], pos=None, initial_pos=None, prev_command: SVGCommand = None):
        if pos is None:
            pos = Point(0.0)
        if initial_pos is None:
            initial_pos = Point(0.0)

        cmd = SVGCmdEnum(cmd_str.lower())

        # Implicit MoveTo commands are treated as LineTo
        if cmd is SVGCmdEnum.MOVE_TO and len(args_str) > 2:
            l_cmd_str = SVGCmdEnum.LINE_TO.value
            if cmd_str.isupper():
                l_cmd_str = l_cmd_str.upper()

            l1, pos, initial_pos = SVGCommand.from_str(cmd_str, args_str[:2], pos, initial_pos)
            l2, pos, initial_pos = SVGCommand.from_str(l_cmd_str, args_str[2:], pos, initial_pos)
            return [*l1, *l2], pos, initial_pos

        nb_args = len(args_str)

        if cmd is SVGCmdEnum.CLOSE_PATH:
            assert nb_args == 0, f"Expected no argument for command {cmd_str}: {nb_args} given"
            return [SVGCommandClose(pos, initial_pos)], initial_pos, initial_pos

        expected_nb_args = sum([ArgType.num_args for ArgType in svgCmdArgTypes[cmd.value]])
        assert (
            nb_args % expected_nb_args == 0
        ), f"Expected {expected_nb_args} arguments for command {cmd_str}: {nb_args} given"

        ls = []
        i = 0
        for _ in range(nb_args // expected_nb_args):
            args = []
            for ArgType in svgCmdArgTypes[cmd.value]:
                num_args = ArgType.num_args
                arg = ArgType(*args_str[i : i + num_args])

                if cmd_str.islower():
                    arg.translate(pos)
                args.append(arg)
                i += num_args

            cmd_parsed = None
            if cmd is SVGCmdEnum.LINE_TO:
                cmd_parsed = SVGCommandLine(pos, *args)
            elif cmd is SVGCmdEnum.MOVE_TO:
                cmd_parsed = SVGCommandMove(pos, *args)
            # elif cmd is SVGCmdEnum.ELLIPTIC_ARC:
            #     cmd_parsed = SVGCommandArc(pos, *args)
            elif cmd is SVGCmdEnum.CUBIC_BEZIER:
                cmd_parsed = SVGCommandBezier(pos, *args)

            pos = cmd_parsed.end_pos

            if cmd is SVGCmdEnum.MOVE_TO:
                initial_pos = pos

            ls.append(cmd_parsed)

        return ls, pos, initial_pos

    def __repr__(self):
        cmd = self.command.value.upper()
        return f"{cmd}{self.get_geoms()}"

    def to_str(self):
        cmd = self.command.value.upper()
        return f"{cmd}{' '.join([arg.to_str() for arg in self.args])}"

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        cmd_index, args = int(vector[0]), vector[1:]

        cmd = SVGCmdEnum(SVGTensor.COMMANDS_SIMPLIFIED[cmd_index])
        start_pos = Point(*args[:2].tolist())
        control1 = Point(*args[2:4].tolist())
        control2 = Point(*args[4:6].tolist())
        end_pos = Point(*args[6:].tolist())

        return SVGCommand.from_args(cmd, start_pos, control1, control2, end_pos)

    @staticmethod
    def from_args(
        command: SVGCmdEnum,
        start_pos: Point,
        control1: Point,
        control2: Point,
        end_pos: Point,
    ):
        if command is SVGCmdEnum.MOVE_TO:
            return SVGCommandMove(start_pos, end_pos)
        elif command is SVGCmdEnum.LINE_TO:
            return SVGCommandLine(start_pos, end_pos)
        elif command is SVGCmdEnum.CUBIC_BEZIER:
            return SVGCommandBezier(start_pos, control1, control2, end_pos)
        elif command is SVGCmdEnum.CLOSE_PATH:
            return SVGCommandClose(start_pos, end_pos)

    def get_geoms(self):
        return [self.start_pos, self.end_pos]

    def sample_points(self, n=10, return_array=False):
        return []

    def get_points_viz(self, first=False, last=False):
        from .svg_primitive import SVGCircle

        color = "red" if first else "purple" if last else "deepskyblue"  # "#C4C4C4"
        opacity = 0.75 if first or last else 1.0
        return [SVGCircle(self.end_pos, radius=Radius(0.4), color=color, fill=True, stroke_width=".1", opacity=opacity)]

    def get_handles_viz(self):
        return []


class SVGCommandLinear(SVGCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tensor(self, PAD_VAL=-1):
        cmd_index = SVGTensor.COMMANDS_SIMPLIFIED.index(self.command.value)
        return torch.tensor([cmd_index, *self.start_pos.to_tensor(), *([PAD_VAL] * 4), *self.end_pos.to_tensor()])

    def numericalize(self, n=256, round_coords=True):
        self.start_pos.numericalize(n, round_coords)
        self.end_pos.numericalize(n, round_coords)

    def copy(self):
        return self.__class__(self.start_pos.copy(), self.end_pos.copy())

    def reverse(self):
        return self.__class__(self.end_pos, self.start_pos)

    def split(self, n=2):
        return [self]

    def bbox(self):
        return Bbox(self.start_pos, self.end_pos)


class SVGCommandMove(SVGCommandLinear):
    def __init__(self, start_pos: Point, end_pos: Point = None):
        if end_pos is None:
            start_pos, end_pos = Point(0.0), start_pos
        super().__init__(SVGCmdEnum.MOVE_TO, [end_pos], start_pos, end_pos)

    def get_points_viz(self, first=False, last=False):
        from .svg_primitive import SVGLine

        points_viz = super().get_points_viz(first, last)
        points_viz.append(SVGLine(self.start_pos, self.end_pos, color="red", dasharray=0.5))
        return points_viz

    def bbox(self):
        return Bbox(self.end_pos, self.end_pos)


class SVGCommandLine(SVGCommandLinear):
    def __init__(self, start_pos: Point, end_pos: Point):
        super().__init__(SVGCmdEnum.LINE_TO, [end_pos], start_pos, end_pos)

    def sample_points(self, n=10, return_array=False):
        z = np.linspace(0.0, 1.0, n)

        if return_array:
            points = (1 - z)[:, None] * self.start_pos.pos[None] + z[:, None] * self.end_pos.pos[None]
            return points

        points = [(1 - alpha) * self.start_pos + alpha * self.end_pos for alpha in z]
        return points

    def split(self, n=2):
        points = self.sample_points(n + 1)
        return [SVGCommandLine(p1, p2) for p1, p2 in zip(points[:-1], points[1:])]

    def length(self):
        return self.start_pos.dist(self.end_pos)


class SVGCommandClose(SVGCommandLinear):
    def __init__(self, start_pos: Point, end_pos: Point):
        super().__init__(SVGCmdEnum.CLOSE_PATH, [], start_pos, end_pos)

    def get_points_viz(self, first=False, last=False):
        return []


class SVGCommandBezier(SVGCommand):
    def __init__(self, start_pos: Point, control1: Point, control2: Point, end_pos: Point):
        if control2 is None:
            control2 = control1.copy()
        super().__init__(SVGCmdEnum.CUBIC_BEZIER, [control1, control2, end_pos], start_pos, end_pos)

        self.control1 = control1
        self.control2 = control2

    @property
    def p1(self):
        return self.start_pos

    @property
    def p2(self):
        return self.end_pos

    @property
    def q1(self):
        return self.control1

    @property
    def q2(self):
        return self.control2

    def copy(self):
        return SVGCommandBezier(self.start_pos.copy(), self.control1.copy(), self.control2.copy(), self.end_pos.copy())

    def to_tensor(self, PAD_VAL=-1):
        cmd_index = SVGTensor.COMMANDS_SIMPLIFIED.index(SVGCmdEnum.CUBIC_BEZIER.value)
        return torch.tensor(
            [
                cmd_index,
                *self.start_pos.to_tensor(),
                *self.control1.to_tensor(),
                *self.control2.to_tensor(),
                *self.end_pos.to_tensor(),
            ]
        )

    def to_vector(self):
        return np.array(
            [self.start_pos.tolist(), self.control1.tolist(), self.control2.tolist(), self.end_pos.tolist()]
        )

    @staticmethod
    def from_vector(vector):
        return SVGCommandBezier(Point(vector[0]), Point(vector[1]), Point(vector[2]), Point(vector[3]))

    def reverse(self):
        return SVGCommandBezier(self.end_pos, self.control2, self.control1, self.start_pos)

    def numericalize(self, n=256, round_coords=True):
        self.start_pos.numericalize(n, round_coords)
        self.control1.numericalize(n, round_coords)
        self.control2.numericalize(n, round_coords)
        self.end_pos.numericalize(n, round_coords)

    def get_geoms(self):
        return [self.start_pos, self.control1, self.control2, self.end_pos]

    def eval(self, t):
        return (
            (1 - t) ** 3 * self.start_pos
            + 3 * (1 - t) ** 2 * t * self.control1
            + 3 * (1 - t) * t**2 * self.control2
            + t**3 * self.end_pos
        )

    def derivative(self, t, n=1):
        if n == 1:
            return (
                3 * (1 - t) ** 2 * (self.control1 - self.start_pos)
                + 6 * (1 - t) * t * (self.control2 - self.control1)
                + 3 * t**2 * (self.end_pos - self.control2)
            )
        elif n == 2:
            return 6 * (1 - t) * (self.control2 - 2 * self.control1 + self.start_pos) + 6 * t * (
                self.end_pos - 2 * self.control2 + self.control1
            )

        raise NotImplementedError

    def angle(self, other: SVGCommandBezier):
        t1, t2 = self.derivative(1.0), -other.derivative(0.0)
        if np.isclose(t1.norm(), 0.0) or np.isclose(t2.norm(), 0.0):
            return 0.0
        angle = np.arccos(np.clip(t1.normalize().dot(t2.normalize()), -1.0, 1.0))
        return np.rad2deg(angle)

    def sample_points(self, n=10, return_array=False):
        b = self.to_vector()

        z = np.linspace(0.0, 1.0, n)
        Z = np.stack([np.ones_like(z), z, z**2, z**3], axis=1)
        Q = np.array([[1.0, 0.0, 0.0, 0.0], [-3, 3.0, 0.0, 0.0], [3.0, -6, 3.0, 0.0], [-1, 3.0, -3, 1]])

        points = Z @ Q @ b

        if return_array:
            return points

        return [Point(p) for p in points]

    def _split_two(self, z=0.5):
        b = self.to_vector()

        Q1 = np.array(
            [
                [1, 0, 0, 0],
                [-(z - 1), z, 0, 0],
                [(z - 1) ** 2, -2 * (z - 1) * z, z**2, 0],
                [-((z - 1) ** 3), 3 * (z - 1) ** 2 * z, -3 * (z - 1) * z**2, z**3],
            ]
        )
        Q2 = np.array(
            [
                [-((z - 1) ** 3), 3 * (z - 1) ** 2 * z, -3 * (z - 1) * z**2, z**3],
                [0, (z - 1) ** 2, -2 * (z - 1) * z, z**2],
                [0, 0, -(z - 1), z],
                [0, 0, 0, 1],
            ]
        )

        return SVGCommandBezier.from_vector(Q1 @ b), SVGCommandBezier.from_vector(Q2 @ b)

    def split(self, n=2):
        b_list = []
        b = self

        for i in range(n - 1):
            z = 1.0 / (n - i)
            b1, b = b._split_two(z)
            b_list.append(b1)
        b_list.append(b)
        return b_list

    def length(self):
        p = self.sample_points(n=100, return_array=True)
        return np.linalg.norm(p[1:] - p[:-1], axis=-1).sum()

    def bbox(self):
        return Bbox.from_points(self.find_extrema())

    def find_roots(self):
        a = 3 * (-self.p1 + 3 * self.q1 - 3 * self.q2 + self.p2)
        b = 6 * (self.p1 - 2 * self.q1 + self.q2)
        c = 3 * (self.q1 - self.p1)

        x_roots, y_roots = get_roots(a.x, b.x, c.x), get_roots(a.y, b.y, c.y)
        roots_cat = [*x_roots, *y_roots]
        roots = [root for root in roots_cat if 0 <= root <= 1]
        return roots

    def find_extrema(self):
        points = [self.start_pos, self.end_pos]
        points.extend([self.eval(root) for root in self.find_roots()])
        return points

    def get_handles_viz(self):
        from .svg_primitive import SVGLine

        return [
            SVGLine(self.start_pos, self.control1, color="#118ab2", dasharray=0.5, opacity=0.7),
            SVGLine(self.control2, self.end_pos, color="#118ab2", dasharray=0.5, opacity=0.7),
        ]

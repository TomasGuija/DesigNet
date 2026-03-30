from __future__ import annotations

from typing import List, Union

import numpy as np
import torch

Num = Union[int, float]
float_type = (int, float, np.float32)


def union_bbox(bbox_list: List[Bbox]):
    res = None
    for bbox in bbox_list:
        if bbox is None:
            continue
        res = bbox.union(res)
    return res


class Geom:
    def copy(self):
        raise NotImplementedError

    def to_str(self):
        raise NotImplementedError

    def to_tensor(self):
        raise NotImplementedError

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        raise NotImplementedError

    def scale(self, factor):
        pass

    def translate(self, vec):
        pass

    def numericalize(self, n=256):
        raise NotImplementedError


class Point(Geom):
    num_args = 2

    def __init__(self, x=None, y=None):
        if isinstance(x, np.ndarray):
            self.pos = x.astype(np.float32)
        elif x is None and y is None:
            self.pos = np.array([0.0, 0.0], dtype=np.float32)
        elif (isinstance(x, float_type) or x is None) and (isinstance(y, float_type) or y is None):
            if x is None:
                x = y
            if y is None:
                y = x
            self.pos = np.array([x, y], dtype=np.float32)
        else:
            raise ValueError()

    def copy(self):
        return Point(self.pos.copy())

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def __add__(self, other):
        return Point(self.pos + other.pos)

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, lmbda):
        if isinstance(lmbda, Point):
            return Point(self.pos * lmbda.pos)

        assert isinstance(lmbda, float_type)
        return Point(lmbda * self.pos)

    def __rmul__(self, lmbda):
        return self * lmbda

    def __truediv__(self, lmbda):
        if isinstance(lmbda, Point):
            return Point(self.pos / lmbda.pos)

        assert isinstance(lmbda, float_type)
        return self * (1 / lmbda)

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"P({self.x}, {self.y})"

    def to_str(self):
        return f"{self.x} {self.y}"

    def tolist(self):
        return self.pos.tolist()

    def to_tensor(self):
        return torch.tensor(self.pos)

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        return Point(*vector.tolist())

    def translate(self, vec: Point):
        self.pos += vec.pos

    def scale(self, factor):
        self.pos *= factor

    def dot(self, other: Point):
        return self.pos.dot(other.pos)

    def norm(self):
        return float(np.linalg.norm(self.pos))

    def dist(self, other: Point):
        return (self - other).norm()

    def normalize(self):
        n = self.norm()
        if np.isclose(n, 0.0):
            return Point(0.0)
        return self / n

    def numericalize(self, n=256, round_coords=True):
        if round_coords:
            self.pos = self.pos.round().clip(min=0, max=n - 1)
        else:
            self.pos = self.pos.clip(min=0, max=n - 1)

    def isclose(self, other: Point):
        return np.allclose(self.pos, other.pos)

    def iszero(self):
        return np.all(self.pos == 0)

    def pointwise_min(self, other: Point):
        return Point(min(self.x, other.x), min(self.y, other.y))

    def pointwise_max(self, other: Point):
        return Point(max(self.x, other.x), max(self.y, other.y))


class Size(Point):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return Size(self.pos.copy())

    def __repr__(self):
        return f"Size({self.pos[0]}, {self.pos[1]})"

    def max(self):
        return self.pos.max()

    def min(self):
        return self.pos.min()

    def translate(self, vec: Point):
        pass


class Radius(Point):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return Radius(self.pos.copy())

    def __repr__(self):
        return f"Rad({self.pos[0]}, {self.pos[1]})"

    def translate(self, vec: Point):
        pass


class Bbox(Geom):
    num_args = 4

    def __init__(self, x=None, y=None, w=None, h=None):
        if isinstance(x, Point) and isinstance(y, Point):
            self.xy = x
            wh = y - x
            self.wh = Size(wh.x, wh.y)
        elif (isinstance(x, float_type) or x is None) and (isinstance(y, float_type) or y is None):
            if x is None:
                x = 0.0
            if y is None:
                y = float(x)

            if w is None and h is None:
                w, h = float(x), float(y)
                x, y = 0.0, 0.0
            self.xy = Point(x, y)
            self.wh = Size(w, h)
        else:
            raise ValueError()

    @property
    def xy2(self):
        return self.xy + self.wh

    def copy(self):
        bbox = Bbox()
        bbox.xy = self.xy.copy()
        bbox.wh = self.wh.copy()
        return bbox

    @property
    def size(self):
        return self.wh

    @property
    def center(self):
        return self.xy + self.wh / 2

    def __repr__(self):
        return f"Bbox({self.xy.to_str()} {self.wh.to_str()})"

    def to_str(self):
        return f"{self.xy.to_str()} {self.wh.to_str()}"

    def to_tensor(self):
        return torch.tensor([*self.xy.to_tensor(), *self.wh.to_tensor()])

    def translate(self, vec):
        self.xy.translate(vec)

    def scale(self, factor):
        self.xy.scale(factor)
        self.wh.scale(factor)

    def union(self, other: Bbox):
        if other is None:
            return self
        return Bbox(self.xy.pointwise_min(other.xy), self.xy2.pointwise_max(other.xy2))

    def intersect(self, other: Bbox):
        if other is None:
            return self

        bbox = Bbox(self.xy.pointwise_max(other.xy), self.xy2.pointwise_min(other.xy2))
        if bbox.wh.x < 0 or bbox.wh.y < 0:
            return None

        return bbox

    @staticmethod
    def from_points(points: List[Point]):
        if not points:
            return None
        xy = xy2 = points[0]
        for p in points[1:]:
            xy = xy.pointwise_min(p)
            xy2 = xy2.pointwise_max(p)
        return Bbox(xy, xy2)

    def area(self):
        return self.wh.pos.prod()

    def overlap(self, other):
        inter = self.intersect(other)
        if inter is None:
            return 0.0
        return inter.area() / self.area()

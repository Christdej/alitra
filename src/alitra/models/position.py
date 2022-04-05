from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .frame import Frame


@dataclass
class Position:
    """
    Position contains the x, y and z coordinate as well as a frame
    """

    x: float
    y: float
    z: float
    frame: Frame

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        if (
            np.allclose(a=[self.x, self.y, self.z], b=[other.x, other.y, other.z])
            and self.frame == other.frame
        ):
            return True
        return False

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(position: np.ndarray, frame: Frame) -> Position:
        if position.shape != (3,):
            raise ValueError("Quaternion array must have shape (4,)")
        return Position(x=position[0], y=position[1], z=position[2], frame=frame)


@dataclass
class Positions:
    """
    Positions contains a list of positions as well as a frame in which the points
    are valid
    """

    positions: List[Position]
    frame: Frame

    def to_array(self) -> np.ndarray:
        positions = []
        for position in self.positions:
            positions.append([position.x, position.y, position.z])
        return np.array(positions, dtype=float)

    @staticmethod
    def from_array(position_array: np.ndarray, frame: Frame) -> Positions:
        if position_array.shape[1] != 3:
            raise ValueError("position_array should have shape (3,N)")
        positions: List[Position] = []
        for position in position_array:
            positions.append(
                Position(x=position[0], y=position[1], z=position[2], frame=frame)
            )
        return Positions(positions=positions, frame=frame)

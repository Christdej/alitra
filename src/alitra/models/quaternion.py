from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import Frame


@dataclass
class Quaternion:
    """
    Quaternion rotation in three dimensions with a frame
    """

    frame: Frame
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=float)

    @staticmethod
    def from_array(
        quaternion: np.ndarray,
        frame: Frame,
    ) -> Quaternion:
        if quaternion.shape != (4,):
            raise ValueError("Quaternion array must have shape (4,)")
        return Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
            frame=frame,
        )

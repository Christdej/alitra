from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import Frame


@dataclass
class Euler:
    """
    Euler angles where (psi,theta,phi) is rotation about the z-,y-, and x- axis respectively to_array and from_array
    are both ordered by rotation about z,y,x. That is [psi,theta,phi].
    """

    frame: Frame
    psi: float = 0
    theta: float = 0
    phi: float = 0

    def to_array(self) -> np.ndarray:
        return np.array([self.psi, self.theta, self.phi], dtype=float)

    @staticmethod
    def from_array(
        rotations: np.ndarray,
        frame: Frame,
    ) -> Euler:
        if rotations.shape != (3,):
            raise ValueError("Coordinate_list should have shape (3,)")
        return Euler(
            psi=rotations[0], theta=rotations[1], phi=rotations[2], frame=frame
        )

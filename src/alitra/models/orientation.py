from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform.rotation import Rotation

from .frame import Frame


@dataclass
class Orientation:
    """
    This class represents an orientation scipy rotation object and a frame
    """

    frame: Frame
    rotation: Rotation

    def to_euler_array(
        self, degrees: bool = False, wrap_angles: bool = False, seq: str = "ZYX"
    ) -> np.ndarray:
        """
        Retrieve the orientation as yaw, pitch, roll euler coordinates. This function
        uses the ZYX intrinsic rotations as default convention.
        :param degrees: Set to true to retrieve angles as degrees.
        :return: List of euler angles [yaw, pitch, roll]
        """

        euler = self.rotation.as_euler(seq=seq, degrees=degrees)

        if wrap_angles:
            base = 360.0 if degrees else 2 * np.pi
            euler = np.array(
                map(lambda angle: ((angle + base) % (base)), euler), dtype=float
            )

        return euler

    def to_quat_array(self) -> np.ndarray:
        return self.rotation.as_quat()

    @staticmethod
    def from_quat_array(quat: np.ndarray, frame: Frame) -> Orientation:
        if quat.shape != (4,):
            raise ValueError("quaternion should have shape (4,)")
        return Orientation(
            frame=frame,
            rotation=Rotation.from_quat(quat),
        )

    @staticmethod
    def from_euler_array(
        euler: np.ndarray, frame: Frame, degrees: bool = False, seq: str = "ZYX"
    ) -> Orientation:
        rotation = Rotation.from_euler(seq=seq, angles=euler, degrees=degrees)
        orientation: Orientation = Orientation(rotation=rotation, frame=frame)
        return orientation

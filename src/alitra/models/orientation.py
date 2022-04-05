from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform.rotation import Rotation

from alitra.convert import quaternion_to_euler

from .euler import Euler
from .frame import Frame
from .quaternion import Quaternion


@dataclass
class Orientation:
    """
    This class represents an orientation using quaternions. Methods that utilize
    Euler angles will all follow the yaw, pitch, roll convention which rotates around
    the ZYX axis with intrinsic rotations.
    """

    x: float
    y: float
    z: float
    w: float
    frame: Frame

    def __eq__(self, other):
        if not isinstance(other, Orientation):
            return False
        if (
            np.allclose(
                a=[self.x, self.y, self.z, self.w],
                b=[other.x, other.y, other.z, other.w],
                atol=1e-10,
            )
            and self.frame == other.frame
        ):
            return True
        return False

    def to_euler_array(
        self,
        degrees: bool = False,
        wrap_angles: bool = False,
    ) -> np.ndarray:
        """
        Retrieve the orientation as yaw, pitch, roll Euler coordinates. This function uses the ZYX intrinsic rotations
        as standard convention.
        :param degrees: Set to true to retrieve angles as degrees.
        :return: List of euler angles [yaw, pitch, roll]
        """

        euler = quaternion_to_euler(
            Quaternion(x=self.x, y=self.y, z=self.z, w=self.w, frame=self.frame),
            sequence="ZYX",
            degrees=degrees,
        ).to_array()

        if wrap_angles:
            base = 360.0 if degrees else 2 * np.pi
            euler = np.array(
                map(lambda angle: ((angle + base) % (base)), euler), dtype=float
            )

        return euler

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=float)

    @staticmethod
    def from_array(orientation: np.ndarray, frame: Frame) -> Orientation:
        if orientation.shape != (4,):
            raise ValueError("orientation should have shape (4,)")
        return Orientation(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
            frame=frame,
        )

    @staticmethod
    def from_euler(euler: Euler, frame: Frame, seq="ZYX") -> Orientation:
        rotation_object = Rotation.from_euler(seq, euler.to_array())
        quaternion: Quaternion = Quaternion.from_array(
            rotation_object.as_quat(), frame=frame
        )
        orientation: Orientation = Orientation.from_array(quaternion.to_array(), frame)
        return orientation

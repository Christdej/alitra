from __future__ import annotations

from dataclasses import dataclass

from scipy.spatial.transform import Rotation

from .euler import Euler
from .frame import Frame
from .quaternion import Quaternion
from .translation import Translation


@dataclass
class Transform:
    """
    A transform object that describe the transformation between two frames.
    Euler or quaternion must be provided to perform a rotation. If no rotation is
    required specify the unit quaternion or zero Euler angles. Translations must be
    expressed in the (to_) frame
    """

    translation: Translation
    from_: Frame
    to_: Frame
    rotation_object: Rotation = None

    def __post_init__(self):
        if (
            not self.translation.from_ == self.from_
            and not self.translation.to_ == self.to_
        ):
            raise ValueError(
                f"The from_ frames or to_ frames of translation and transform object are not equal."
            )

    @staticmethod
    def from_euler(
        translation: Translation, euler: Euler, from_: Frame, to_: Frame, seq="ZYX"
    ) -> Transform:
        rotation_object = Rotation.from_euler(seq=seq, angles=euler.to_array())
        return Transform(
            translation=translation,
            from_=from_,
            to_=to_,
            rotation_object=rotation_object,
        )

    @staticmethod
    def from_quat(
        translation: Translation,
        quat: Quaternion,
        from_: Frame,
        to_: Frame,
    ) -> Transform:
        rotation_object = Rotation.from_quat(quat.to_array())
        return Transform(
            translation=translation,
            from_=from_,
            to_=to_,
            rotation_object=rotation_object,
        )

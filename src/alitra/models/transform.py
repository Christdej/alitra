from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .frame import Frame
from .translation import Translation


@dataclass
class Transform:
    """
    A transform object that describe the transformation between two frames.
    Contains a scipy rotation object a translation and two frames.
    Can be created from euler array or quaternion array. Translations must be
    expressed in the (to_) frame
    """

    translation: Translation
    from_: Frame
    to_: Frame
    rotation: Rotation = None

    def __post_init__(self):
        if (
            not self.translation.from_ == self.from_
            and not self.translation.to_ == self.to_
        ):
            raise ValueError(
                f"The from_ frames or to_ frames of translation and transform object are not equal."
            )

    @staticmethod
    def from_euler_array(
        translation: Translation, euler: np.ndarray, from_: Frame, to_: Frame, seq="ZYX"
    ) -> Transform:
        rotation = Rotation.from_euler(seq=seq, angles=euler)
        return Transform(
            translation=translation,
            from_=from_,
            to_=to_,
            rotation=rotation,
        )

    @staticmethod
    def from_quat_array(
        translation: Translation,
        quat: np.ndarray,
        from_: Frame,
        to_: Frame,
    ) -> Transform:
        rotation = Rotation.from_quat(quat)
        return Transform(
            translation=translation,
            from_=from_,
            to_=to_,
            rotation=rotation,
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alitra.models import Frame


@dataclass
class Translation:
    """Translations should be expressed in the to_ frame, which are typically the asset frame"""

    x: float
    y: float
    from_: Frame
    to_: Frame
    z: float = 0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(
        translation: np.ndarray,
        from_: Frame,
        to_: Frame,
    ) -> Translation:
        if translation.shape != (3,):
            raise ValueError("translation should have shape (3,)")
        return Translation(
            x=translation[0], y=translation[1], z=translation[2], from_=from_, to_=to_
        )

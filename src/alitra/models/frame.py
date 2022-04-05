from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Frame:
    """
    Frame is used by most of our models to describe in which frame a model lives,
    or the two frames a transform is between.
    """

    name: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return False
        if self.name == other.name:
            return True
        return False

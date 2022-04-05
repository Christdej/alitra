from dataclasses import dataclass

from .frame import Frame
from .orientation import Orientation
from .position import Position


@dataclass
class Pose:
    """
    Pose contains a position, an orientation and a frame
    """

    position: Position
    orientation: Orientation
    frame: Frame

from typing import Union

import numpy as np
from scipy.spatial.transform.rotation import Rotation

from .models.frame import Frame
from .models.orientation import Orientation
from .models.pose import Pose
from .models.position import Position, Positions
from .models.transform import Transform


def transform_position(
    transform: Transform,
    positions: Union[Position, Positions],
    from_: Frame,
    to_: Frame,
) -> Union[Position, Positions]:
    """
    Transforms a position or list of positions from from_ to to_ (rotation and translation)
    :param transform: Transform between two coordinate systems
    :param positions: Position or Positions in the from_ coordinate system.
    :param from_: Source Frame, must be different to "to_".
    :param to_: Destination Frame, must be different to "from_".
    :return: Position or Positions in the to_ coordinate system.
    """
    if positions.frame != from_:
        raise ValueError(
            f"Expected positions in frame {from_} "
            + f", got positions in frame {positions.frame}"
        )

    if from_ == to_:
        return positions

    result: np.ndarray
    if from_ == transform.to_ and to_ == transform.from_:
        """Using the inverse transform"""
        result = transform.rotation.apply(
            positions.to_array() - transform.translation.to_array(),
            inverse=True,
        )
    elif from_ == transform.from_ and to_ == transform.to_:
        result = (
            transform.rotation.apply(positions.to_array())
            + transform.translation.to_array()
        )
    else:
        raise ValueError("Transform not specified")

    if isinstance(positions, Position):
        return Position.from_array(result, to_)
    elif isinstance(positions, Positions):
        return Positions.from_array(result, to_)
    else:
        raise ValueError("Incorrect input format. Must be Position or Positions.")


def transform_rotation(
    transform: Transform, rotation: Rotation, from_: Frame, to_: Frame
) -> Rotation:
    """
    Transforms a rotation from from_ to to_ (rotation)
    :param transform: Transform between two coordinate systems
    :param rotation: Rotation (scipy) in the from_ coordinate system.
    :param from_: Source Frame, must be different to "to_".
    :param to_: Destination Frame, must be different to "from_".
    :return: Rotation in the to_ coordinate system.
    """

    if from_ == transform.to_ and to_ == transform.from_:
        "Using the inverse transform"
        rotation_to = rotation * transform.rotation.inv()
    elif from_ == transform.from_ and to_ == transform.to_:
        rotation_to = rotation * transform.rotation
    else:
        raise ValueError("Transform not specified")

    return rotation_to


def transform_orientation(
    transform: Transform,
    orientation: Orientation,
    from_: Frame,
    to_: Frame,
) -> Orientation:
    """
    Transforms an orientation from from_ to to_ (rotation)
    :param transform: Transform between two coordinate systems
    :param orientation: Orientation in the from_ coordinate system.
    :param from_: Source Frame, must be different to "to_".
    :param to_: Destination Frame, must be different to "from_".
    :return: Orientation in the to_ coordinate system.
    """

    if from_ == to_:
        return orientation

    rotation_to = transform_rotation(transform, orientation.rotation, from_, to_)
    return Orientation(rotation=rotation_to, frame=to_)


def transform_pose(transform: Transform, pose: Pose, from_: Frame, to_: Frame) -> Pose:
    """
    Transforms a pose from from_ to to_ (rotation)
    :param transform: Transform between two coordinate systems
    :param pose: Pose in the from_ coordinate system.
    :param from_: Source Frame, must be different to "to_".
    :param to_: Destination Frame, must be different to "from_".
    :return: Pose in the to_ coordinate system.
    """

    if from_ == to_:
        return pose

    position = transform_position(transform, pose.position, from_, to_)
    if not isinstance(position, Position):
        raise TypeError("Pose can only contain a single position, not positions")
    orientation: Orientation = transform_orientation(
        transform, pose.orientation, from_, to_
    )

    return Pose(position, orientation, to_)

import math

import numpy as np
import pytest

from alitra import (
    transform_euler,
    transform_orientation,
    transform_pose,
    transform_position,
    transform_quaternion,
)
from alitra.models import (
    Euler,
    Frame,
    Orientation,
    Pose,
    Position,
    Positions,
    Quaternion,
    Transform,
    Translation,
)


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_expected",
    [
        (
            Euler(psi=0.0, frame=Frame("robot")),
            Translation(x=0, y=0, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [1, 2, 3],
                        [-1, -2, -3],
                        [0, 0, 0],
                        [100000, 1, -100000],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
        (
            Euler(psi=np.pi * -0.0, frame=Frame("robot")),
            Translation(x=10, y=10, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [11, 12, 3],
                        [9, 8, -3],
                        [10, 10, 0],
                        [100010, 11, -100000],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
        (
            Euler(psi=np.pi / 2, frame=Frame("robot")),
            Translation(x=10, y=0, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [8, 1, 3],
                        [12, -1, -3],
                        [10, 0, 0],
                        [9, 100000, -100000],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
        (
            Euler(theta=1 * 0.2, phi=1, psi=0.4, frame=Frame("robot")),
            Translation(x=0, y=10, z=2, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [2.06950653e00, 9.30742421e00, 5.03932254e00],
                        [-2.06950653e00, 1.06925758e01, -1.03932254e00],
                        [0.00000000e00, 1.00000000e01, 2.00000000e00],
                        [4.76148230e04, 1.11500688e05, -7.28173316e04],
                    ]
                ),
                frame=Frame("asset"),
            ),
        ),
    ],
)
def test_transform_list_of_positions(eul_rot, ref_translations, p_expected):
    p_robot = Positions.from_array(
        np.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [0, 0, 0],
                [100000, 1, -100000],
            ],
        ),
        frame=Frame("robot"),
    )
    transform = Transform.from_euler(
        euler=eul_rot,
        translation=ref_translations,
        from_=ref_translations.from_,
        to_=ref_translations.to_,
    )

    p_asset = transform_position(
        transform, p_robot, from_=Frame("robot"), to_=Frame("asset")
    )

    assert p_asset.frame == p_expected.frame
    assert np.allclose(p_expected.to_array(), p_asset.to_array())


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_expected",
    [
        (
            Euler(psi=math.pi / 2.0, frame=Frame("robot")),
            Translation(x=1, y=2, from_=Frame("robot"), to_=Frame("asset")),
            Position.from_array(np.array([-1, 3, 3]), frame=Frame("asset")),
        ),
    ],
)
def test_transform_position(eul_rot, ref_translations, p_expected):

    p_robot = Position.from_array(np.array([1, 2, 3]), frame=Frame("robot"))
    transform = Transform.from_euler(
        euler=eul_rot,
        translation=ref_translations,
        from_=ref_translations.from_,
        to_=ref_translations.to_,
    )

    p_asset = transform_position(
        transform, p_robot, from_=Frame("robot"), to_=Frame("asset")
    )

    assert p_asset.frame == p_expected.frame
    assert np.allclose(p_expected.to_array(), p_asset.to_array())


def test_no_transformation_when_equal_frames():
    p_robot = Position.from_array(np.array([1, 2, 3]), frame=Frame("robot"))
    p_expected = Position.from_array(np.array([1, 2, 3]), frame=Frame("robot"))

    transform = Transform.from_euler(
        euler=Euler(psi=1.0, frame=Frame("robot")),
        translation=Translation(x=2, y=3, from_=Frame("robot"), to_=Frame("asset")),
        from_=Frame("robot"),
        to_=Frame("asset"),
    )

    position = transform_position(
        transform, p_robot, from_=Frame("robot"), to_=Frame("robot")
    )

    assert position.frame == p_expected.frame
    assert np.allclose(p_expected.to_array(), position.to_array())


@pytest.mark.parametrize(
    "from_, to_, error_expected",
    [
        (Frame("asset"), Frame("asset"), False),
        (Frame("robot"), Frame("robot"), False),
        (Frame("robot"), Frame("asset"), False),
        (Frame("asset"), Frame("robot"), True),
    ],
)
def test_transform_position_error(from_, to_, error_expected):
    p_robot = Positions.from_array(
        np.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
                [0, 0, 0],
                [100000, 1, -100000],
            ],
        ),
        frame=Frame("robot"),
    )
    eul_rot = Euler(psi=0.0, frame=Frame("robot"))
    translation = Translation(x=0, y=0, from_=Frame("robot"), to_=Frame("asset"))
    transform = Transform.from_euler(
        euler=eul_rot, translation=translation, from_=Frame("robot"), to_=Frame("asset")
    )

    if error_expected:
        with pytest.raises(ValueError):
            transform_position(transform, p_robot, from_=from_, to_=to_)
    else:
        transform_position(transform, p_robot, from_=from_, to_=to_)


@pytest.mark.parametrize(
    "quaternion, rotation_quaternion, expected",
    [
        (
            Quaternion(x=0, y=0, z=0, w=1.0, frame=Frame("robot")),
            Quaternion(x=0, y=0, z=1, w=0, frame=Frame("robot")),
            Quaternion(x=0, y=0, z=1, w=0, frame=Frame("asset")),
        ),
        (
            Quaternion(x=0, y=0, z=0, w=1.0, frame=Frame("asset")),
            Quaternion(x=0, y=0, z=1, w=0, frame=Frame("robot")),
            Quaternion(x=0, y=0, z=1, w=0, frame=Frame("robot")),
        ),
    ],
)
def test_transform_quaternion(quaternion, rotation_quaternion, expected):
    translation: Translation = Translation(
        x=0, y=0, from_=quaternion.frame, to_=expected.frame
    )
    transform = Transform.from_quat(
        quat=rotation_quaternion,
        translation=translation,
        from_=quaternion.frame,
        to_=expected.frame,
    )

    rotated_quaternion: Quaternion = transform_quaternion(
        transform, quaternion=quaternion, from_=quaternion.frame, to_=expected.frame
    )

    assert rotated_quaternion == expected


@pytest.mark.parametrize(
    "euler, rotation_euler, expected",
    [
        (
            Euler(psi=0, theta=0, phi=0, frame=Frame("robot")),
            Euler(psi=0, theta=0, phi=1, frame=Frame("robot")),
            Euler(psi=0, theta=0, phi=1, frame=Frame("asset")),
        ),
        (
            Euler(psi=0, theta=0, phi=0, frame=Frame("asset")),
            Euler(psi=0, theta=0, phi=1, frame=Frame("robot")),
            Euler(psi=0, theta=0, phi=1, frame=Frame("robot")),
        ),
    ],
)
def test_transform_euler(euler, rotation_euler, expected):
    translation: Translation = Translation(
        x=0, y=0, from_=euler.frame, to_=expected.frame
    )
    transform = Transform.from_euler(
        translation=translation,
        euler=rotation_euler,
        from_=euler.frame,
        to_=expected.frame,
    )

    rotated_euler: Euler = transform_euler(
        transform, euler=euler, from_=euler.frame, to_=expected.frame
    )

    assert rotated_euler == expected


def test_transform_orientation(
    default_transform, default_orientation, robot_frame, asset_frame
):
    expected_orientation: Orientation = Orientation(
        x=0, y=0, z=0, w=1, frame=asset_frame
    )

    orientation: Orientation = transform_orientation(
        transform=default_transform,
        orientation=default_orientation,
        from_=robot_frame,
        to_=asset_frame,
    )
    assert np.allclose(expected_orientation.to_array(), orientation.to_array())


def test_transform_pose(default_transform, default_pose, robot_frame, asset_frame):
    expected_pose: Pose = Pose(
        position=Position(x=0, y=0, z=0, frame=asset_frame),
        orientation=Orientation(x=0, y=0, z=0, w=1, frame=asset_frame),
        frame=asset_frame,
    )

    pose_to: Pose = transform_pose(
        transform=default_transform,
        pose=default_pose,
        from_=robot_frame,
        to_=asset_frame,
    )
    assert np.allclose(
        expected_pose.orientation.to_array(), pose_to.orientation.to_array()
    )
    assert np.allclose(expected_pose.position.to_array(), pose_to.position.to_array())
    assert expected_pose.frame == pose_to.frame

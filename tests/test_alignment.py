import numpy as np
import pytest

from alitra import align_maps, align_positions, transform_position
from alitra.models import Euler, Frame, Positions, Transform, Translation


@pytest.mark.parametrize(
    "eul_rot, ref_translations, p_robot,rotation_axes",
    [
        (
            Euler(psi=np.pi * -0.0, frame=Frame("robot")),
            Translation(x=200030, y=10000, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [10, 1, 0],
                        [20, 2, 0],
                        [30, 7, 0],
                        [40, 5, 0],
                    ]
                ),
                frame=Frame("robot"),
            ),
            "z",
        ),
        (
            Euler(psi=np.pi / 2, frame=Frame("robot")),
            Translation(x=10, y=0, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array([[5, 0, 0], [5, 2, 0], [7, 5, 0], [3, 5, 0]]),
                frame=Frame("robot"),
            ),
            "z",
        ),
        (
            Euler(phi=np.pi * 0.9, frame=Frame("robot")),
            Translation(x=1, y=10, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array([[10, 0, 0], [5, 2, 0], [7, 5, 0], [3, 5, 0]]),
                frame=Frame("robot"),
            ),
            "x",
        ),
        (
            Euler(phi=1 * 0.2, theta=1, psi=0.4, frame=Frame("robot")),
            Translation(x=0, y=10, z=2, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array(
                    [
                        [0, 1, 2],
                        [5, 2, 6],
                        [7, 5, 0],
                        [3, 5, 0],
                        [3, 5, 10],
                        [3, 5, 11],
                    ]
                ),
                frame=Frame("robot"),
            ),
            "xyz",
        ),
        (
            Euler(psi=np.pi / 4, frame=Frame("robot")),
            Translation(x=1, y=0, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array([[1, 1, 0], [10, 1, 0]]), frame=Frame("robot")
            ),
            "z",
        ),
        (
            Euler(theta=np.pi * 0.2, frame=Frame("robot")),
            Translation(x=1, y=10, z=2, from_=Frame("robot"), to_=Frame("asset")),
            Positions.from_array(
                np.array([[0, 1, 2], [5, 2, 0], [7, 5, 0], [3, 5, 0]]),
                frame=Frame("robot"),
            ),
            "y",
        ),
    ],
)
def test_align_frames(eul_rot, ref_translations, p_robot, rotation_axes):
    rotations_c2to_c1 = eul_rot.to_array()
    transform = Transform.from_euler(
        euler=eul_rot,
        translation=ref_translations,
        from_=ref_translations.from_,
        to_=ref_translations.to_,
    )
    ref_translation_array = ref_translations.to_array()
    p_asset = transform_position(
        transform, p_robot, from_=Frame("robot"), to_=Frame("asset")
    )
    frame_transform = align_positions(p_robot, p_asset, rotation_axes)

    assert np.allclose(frame_transform.translation.to_array(), ref_translation_array)

    assert np.allclose(
        frame_transform.rotation_object.as_euler("ZYX"), rotations_c2to_c1
    )

    p_robot_noisy = Positions.from_array(
        p_robot.to_array()
        + np.clip(np.random.normal(np.zeros(p_robot.to_array().shape), 0.1), -0.1, 0.1),
        frame=Frame("robot"),
    )

    p_asset_noisy = Positions.from_array(
        p_asset.to_array()
        + np.clip(np.random.normal(np.zeros(p_asset.to_array().shape), 0.1), -0.1, 0.1),
        frame=Frame("asset"),
    )

    transform_noisy = align_positions(p_robot_noisy, p_asset_noisy, rotation_axes)

    translation_arr_noise = transform_noisy.translation.to_array()
    euler_arr_noise = transform_noisy.rotation_object.as_euler("ZYX")
    rotations = np.absolute(euler_arr_noise - rotations_c2to_c1)
    translations = np.absolute(translation_arr_noise - ref_translation_array)
    assert np.any(rotations > 0.3) == False
    assert np.any(translations > 0.4) == False


def test_align_maps(
    default_map, default_map_asset, default_position, robot_frame, asset_frame
):
    transform = align_maps(map_from=default_map, map_to=default_map_asset, rot_axes="z")

    position_to = transform_position(
        transform=transform,
        positions=default_position,
        from_=robot_frame,
        to_=asset_frame,
    )

    assert np.allclose(default_position.to_array(), position_to.to_array())


@pytest.mark.parametrize(
    "p_asset, p_robot, rotation_frame",
    [
        (
            Positions.from_array(
                np.array([[10, 0, 0], [5, 2, 0], [7, 5, 0]]), Frame("asset")
            ),
            Positions.from_array(
                np.array([[12, 0, 0], [5, 2, 0], [7, 5, 0]]), Frame("robot")
            ),
            "z",
        ),
        (
            Positions.from_array(np.array([[10, 0, 0], [5, 2, 0]]), Frame("asset")),
            Positions.from_array(np.array([[13, 2, 0], [7, 4, 0]]), Frame("robot")),
            "z",
        ),
        ## TODO: Investigate why this is not failing anymore
        # (
        #     Positions.from_array(np.array([[10, 0, 0], [5, 2, 0]]), Frame("robot")),
        #     Positions.from_array(np.array([[11, 0, 0], [6, 2, 0]]), Frame("asset")),
        #     "z",
        # ),
        (
            Positions.from_array(
                np.array([[10, 0, 0], [10, 0, 0], [5, 2, 0], [7, 5, 0]]), Frame("asset")
            ),
            Positions.from_array(
                np.array([[11, 0, 0], [11, 0, 0], [5, 2, 0], [7, 5, 0]]), Frame("robot")
            ),
            "z",
        ),
        (
            Positions.from_array(np.array([[10, 0, 0], [5, 2, 0]]), Frame("asset")),
            Positions.from_array(np.array([[11, 0, 0], [6, 2, 0]]), Frame("robot")),
            "xyz",
        ),
    ],
)
def test_align_frames_exceptions(p_robot, p_asset, rotation_frame):
    with pytest.raises(ValueError):
        align_positions(
            positions_from=p_robot, positions_to=p_asset, rot_axes=rotation_frame
        )

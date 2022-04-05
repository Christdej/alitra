from pathlib import Path

import pytest

from alitra.models import (
    Bounds,
    Euler,
    Frame,
    Map,
    Orientation,
    Pose,
    Position,
    Positions,
    Quaternion,
    Transform,
    Translation,
)


@pytest.fixture()
def robot_frame():
    return Frame("robot")


@pytest.fixture()
def asset_frame():
    return Frame("asset")


@pytest.fixture()
def default_position(robot_frame):
    return Position(x=0, y=0, z=0, frame=robot_frame)


@pytest.fixture()
def robot_position_1(robot_frame):
    return Position(x=0, y=0, z=0, frame=robot_frame)


@pytest.fixture()
def robot_position_2(robot_frame):
    return Position(x=1, y=1, z=1, frame=robot_frame)


def robot_positions(robot_position_1, robot_position_2, robot_frame):
    return Positions([robot_position_1, robot_position_2], frame=robot_frame)


@pytest.fixture()
def default_orientation(robot_frame):
    return Orientation(x=0, y=0, z=0, w=1, frame=robot_frame)


@pytest.fixture()
def default_pose(default_position, default_orientation, robot_frame):
    return Pose(
        position=default_position, orientation=default_orientation, frame=robot_frame
    )


@pytest.fixture()
def default_translation(robot_frame, asset_frame):
    return Translation(x=0, y=0, z=0, from_=robot_frame, to_=asset_frame)


@pytest.fixture()
def default_euler(robot_frame):
    return Euler(psi=0, theta=0, phi=0, frame=robot_frame)


@pytest.fixture()
def default_quaternion(robot_frame):
    return Quaternion(x=0, y=0, z=0, w=1, frame=robot_frame)


@pytest.fixture()
def default_transform(
    default_translation, default_quaternion, robot_frame, asset_frame
):
    return Transform.from_quat(
        translation=default_translation,
        quat=default_quaternion,
        from_=robot_frame,
        to_=asset_frame,
    )


@pytest.fixture()
def default_bounds(robot_position_1, robot_position_2):
    return Bounds(robot_position_1, robot_position_2)


@pytest.fixture()
def default_map():
    here = Path(__file__).parent.resolve()
    map_path = Path(here.joinpath("./test_data/test_map_config.json"))
    return Map.from_config(map_path)


@pytest.fixture()
def default_map_asset():
    here = Path(__file__).parent.resolve()
    map_path = Path(here.joinpath("./test_data/test_map_config_asset.json"))
    return Map.from_config(map_path)


@pytest.fixture()
def default_map_bounds():
    here = Path(__file__).parent.resolve()
    map_path = Path(here.joinpath("./test_data/test_map_config_bounds.json"))
    return Map.from_config(map_path)

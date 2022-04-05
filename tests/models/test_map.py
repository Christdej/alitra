from pathlib import Path

import pytest

from alitra.models import Bounds, Frame, Map, Position, Positions

robot_frame = Frame("robot")

expected_map = Map(
    name="test_map",
    reference_positions=Positions(
        positions=[
            Position(x=10, y=20, z=30, frame=robot_frame),
            Position(x=40, y=50, z=60, frame=robot_frame),
            Position(x=70, y=80, z=90, frame=robot_frame),
        ],
        frame=robot_frame,
    ),
    frame=robot_frame,
)

expected_map_bounds = Map(
    name="test_map_bounds",
    reference_positions=Positions(
        positions=[
            Position(x=10, y=20, z=30, frame=robot_frame),
            Position(x=40, y=50, z=60, frame=robot_frame),
            Position(x=70, y=80, z=90, frame=robot_frame),
        ],
        frame=robot_frame,
    ),
    frame=robot_frame,
    bounds=Bounds(
        position1=Position(x=5, y=15, z=30, frame=robot_frame),
        position2=Position(x=80, y=90, z=100, frame=robot_frame),
    ),
)


def test_load_map():
    map_path = Path("./tests/test_data/test_map_config.json")
    map: Map = Map.from_config(map_path)
    assert map == expected_map


def test_invalid_file_path():
    map_path = Path("./tests/test_data/no_file.json")
    with pytest.raises(Exception):
        Map.from_config(map_path)


def test_load_map_bounds():
    map_path = Path("./tests/test_data/test_map_config_bounds.json")
    map: Map = Map.from_config(map_path)
    assert map == expected_map_bounds
    assert map.bounds.x_min == expected_map_bounds.bounds.position1.x
    assert map.bounds.x_min == expected_map_bounds.bounds.position1.x
    assert map.bounds.y_min == expected_map_bounds.bounds.position1.y
    assert map.bounds.y_max == expected_map_bounds.bounds.position2.y
    assert map.bounds.z_max == expected_map_bounds.bounds.position2.z
    assert map.bounds.z_max == expected_map_bounds.bounds.position2.z

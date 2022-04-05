import pytest

from alitra.models import Position


def test_position_within_bounds(default_bounds, robot_frame):
    pos = Position(0.5, 0.5, 0.5, robot_frame)
    assert default_bounds.position_within_bounds(pos) == True


def test_position_outside_bounds(default_bounds, robot_frame):
    pos = Position(11, 11, 11, robot_frame)
    assert default_bounds.position_within_bounds(pos) == False


def test_position_wrong_frame(default_bounds, asset_frame):
    pos = Position(0.5, 0.5, 0.5, asset_frame)
    with pytest.raises(TypeError):
        default_bounds.position_within_bounds(pos)

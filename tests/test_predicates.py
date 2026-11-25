"""Task relations over typed observation nodes: booleans and gaps agree, boxes turn with their entity."""
import math

import pytest
import torch

from shared.assembly.objects import ObjectView
from shared.predicates import (bottom, distance, inside, inside_gap, is_closed, is_open, near, on_top, on_top_gap,
                               open_fraction, top, upright)

TURN = [math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4)]  # 90 degrees about z


def node(name, **fields):
    values = {f'{name}.{field}': torch.as_tensor(value, dtype=torch.float32) for field, value in fields.items()}
    return getattr(ObjectView(values), name)


def entity(name, position, size, orientation=(1., 0., 0., 0.), offset=(0., 0., 0.)):
    """A body whose box is ``size`` around ``offset`` in its own frame, two environments."""
    half = torch.tensor(size) / 2
    bounds = torch.stack((torch.tensor(offset) - half, torch.tensor(offset) + half))
    return node(name, position=[position, position], orientation=[orientation, orientation],
                bounds=bounds.expand(2, 2, 3))


def test_inside_uses_the_turned_box_and_gap_is_zero_exactly_when_inside():
    oven = entity('oven', (1., 0., 0.), (.4, .2, .2), TURN, offset=(0., 0., .1))
    bowl = node('bowl', position=[[1., .15, .1], [1.3, 0., .1]])
    # Turned 90 degrees, the oven is 0.2 m wide along x and 0.4 m along y.
    assert inside(bowl, oven).tolist() == [True, False]
    assert inside_gap(bowl, oven)[0] == 0 and inside_gap(bowl, oven)[1] == pytest.approx(.2)
    assert inside(bowl, oven, margin=.06).tolist() == [False, False]


def test_on_top_needs_the_footprint_and_the_lowest_point_at_the_top():
    table = entity('table', (0., 0., .375), (1., 1., .75))
    cup = entity('cup', (0., 0., .75), (.1, .1, .1), offset=(0., 0., .05))
    assert bottom(cup)[0] == pytest.approx(.75) and top(table)[0] == pytest.approx(.75)
    assert on_top(cup, table).all() and (on_top_gap(cup, table) == 0).all()
    lifted = entity('cup', (0., 0., .9), (.1, .1, .1))
    beside = entity('cup', (.8, 0., .75), (.1, .1, .1), offset=(0., 0., .05))
    assert not on_top(lifted, table).any() and on_top_gap(lifted, table)[0] == pytest.approx(.85 - .75 - .02)
    assert not on_top(beside, table).any() and on_top_gap(beside, table)[0] == pytest.approx(.3)


def test_abstract_rectangle_reaches_without_end_and_points_have_no_extent():
    area = node('area', position=[[0., 0., 0.]], orientation=[[1., 0., 0., 0.]], size=[[1., 1.]])
    high = node('drone', position=[[.2, .2, 5.]])
    assert inside(high, area).item() and top(area).item() == 0
    assert distance(high, torch.tensor([[.2, .2, 4.]])).item() == pytest.approx(1.)
    assert near(high, torch.tensor([[.2, .2, 4.5]]), .6).item()


def test_open_fraction_measures_from_the_limit_nearest_the_default():
    door = node('door', joint_position=[[-1.047], [0.]], default_joint_position=[[0.], [0.]],
                joint_limits=[[[-2.094, 0.]], [[-2.094, 0.]]])
    assert open_fraction(door, 0).tolist() == pytest.approx([.5, 0.], abs=1e-4)
    assert is_open(door, 0, .4).tolist() == [True, False]
    assert is_closed(door).tolist() == [False, True]


def test_upright_reads_the_local_up_axis():
    tipped = node('bowl', orientation=[[1., 0., 0., 0.], [math.cos(math.pi / 4), math.sin(math.pi / 4), 0., 0.]])
    assert upright(tipped, .1).tolist() == [True, False]

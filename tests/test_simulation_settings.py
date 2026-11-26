"""Physics engine / renderer choices the editor writes into a project's train.yaml."""
import pytest

from shared.assembly.editing import simulation_settings, write_simulation_settings

RECIPE = """defaults:
  - state_ppo
  - _self_
env:
  train:
    init_params:
      backend: isaaclab3
      # kept as written
      physics: ovphysx  # chosen for the apple grasp
    total_num_envs: 4
actor:
  model:
    obs_dim: null
"""


def test_choices_come_from_the_backend_and_only_their_lines_change(tmp_path):
    (tmp_path / 'scene.yaml').write_text('schema_version: 1\nname: world\nentities: []\n')
    (tmp_path / 'train.yaml').write_text(RECIPE)
    scene = tmp_path / 'scene.yaml'
    backend, choices, current = simulation_settings(scene)
    assert backend == 'isaaclab3' and set(choices) == {'physics', 'renderer'}
    assert current == {'physics': 'ovphysx', 'renderer': 'newton'}  # renderer: the backend default
    write_simulation_settings(scene, {'physics': 'newton', 'renderer': 'ovrtx'})
    assert (tmp_path / 'train.yaml').read_text() == RECIPE.replace(
        'physics: ovphysx  # chosen', 'physics: newton  # chosen').replace(
        '  # chosen for the apple grasp\n', '  # chosen for the apple grasp\n      renderer: ovrtx\n')
    assert simulation_settings(scene)[2] == {'physics': 'newton', 'renderer': 'ovrtx'}
    with pytest.raises(ValueError, match='physics must be one of'):
        write_simulation_settings(scene, {'physics': 'bullet'})

pytestmark = pytest.mark.group_core  # runs in the [core] dependency group

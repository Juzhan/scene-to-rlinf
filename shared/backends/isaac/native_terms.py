"""Native implementation paths for Isaac Lab's MDP terms.

Only the strings the editor shows as a term's implementation live here: the
shared catalog describes terms and validation without knowing which module or
function a simulator binds them to.
"""

MDP = "isaaclab.envs.mdp."
LIFT = "isaaclab_tasks.manager_based.manipulation.lift.mdp."
VELOCITY = "isaaclab_tasks.manager_based.locomotion.velocity.mdp."
# Entity signals read through the backend's own primitive reader.
DIRECT_SIGNALS = {"position", "default_joint_position", "default_joint_velocity", "ray_hit_positions",
                  "frame_positions", "contact_force", "contact_force_samples"}


def observation(function):
    return LIFT + function if function == "object_position_in_robot_root_frame" else MDP + function


def action(class_name):
    return MDP + class_name


def reward(function):
    if function.startswith("object_"):
        return LIFT + function
    if function == "feet_air_time":
        return VELOCITY + function
    return MDP + function


def signal(signal_name, function):
    if signal_name in DIRECT_SIGNALS:
        return "shared.backends.isaac.entity_observations.primitive"
    return MDP + function

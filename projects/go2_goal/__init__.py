"""Default project: one policy layer walking a Go2 to a goal."""
from shared.assembly.project import Project
project = Project(__file__, __name__, task_id='Workspace-Go2-Goal-v0', model_type='go2_goal_policy')
register = project.register
check = project.check

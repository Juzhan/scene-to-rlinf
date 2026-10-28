"""Default planner/executor project."""
from shared.assembly.project import Project
project = Project(__file__, __name__, task_id='go2_piper_wholebody/Workspace-Planner-Executor-v0', model_type='go2_piper_wholebody__planner_executor_policy')
register = project.register
check = project.check

"""Default planner/executor project."""
from shared.assembly.project import Project
project = Project(__file__, __name__, task_id='Workspace-Planner-Executor-v0', model_type='planner_executor_policy')
register = project.register
check = project.check

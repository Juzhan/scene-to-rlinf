"""Box stacking scene plus a Newton differentiable-simulation experiment."""

from shared.assembly.project import Project

project = Project(__file__, __name__, task_id="Workspace-Box-Stack-Diffsim-v0", model_type="box_stack_diffsim_policy")
register = project.register
check = project.check

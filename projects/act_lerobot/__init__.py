"""ACT trained on a LeRobot dataset through RLinf's supervised (SFT) worker."""
from shared.integrations.rlinf.sft_project import SftProject

project = SftProject(__file__, __name__, model_type='act_lerobot', export_prefix='act')
register = project.register
run_recipe = project.run_recipe

from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto

from nas.utils.utils import project_root

path = project_root() / "_results/debug/master_2/2024-08-01_01-30-21/history.json"

history = OptHistory.load(path)

print(history)

# history.show(PlotTypesEnum.operations_animated_bar)
# history.show(PlotTypesEnum.operations_kde)
history.show(PlotTypesEnum.fitness_line_interactive)

visualise_pareto(
    history.final_choices.data, show=True
)

print(history.get_leaderboard())

chosen = history.final_choices[0].graph
PipelineVisualizer(chosen).visualise()

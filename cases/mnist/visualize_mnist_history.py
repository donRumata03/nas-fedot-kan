from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum

from nas.utils.utils import project_root

path = project_root() / "_results/debug/master_2/2024-07-31_00-17-45/history.json"

history = OptHistory.load(path)

print(history)

# history.show(PlotTypesEnum.operations_animated_bar)
# history.show(PlotTypesEnum.operations_kde)
history.show(PlotTypesEnum.fitness_line_interactive)

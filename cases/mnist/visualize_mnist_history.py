from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt

from nas.utils.utils import project_root

path = project_root() / "_results/debug/master_2/2024-08-07_23-43-52/history.json"

history = OptHistory.load(path)

print(history)

# Analyze if there is a graph in the history that has arity 2 at least for one node:
for generation in history.individuals:
    for ind in generation:
        graph = ind.graph
        for node in graph.nodes:
            if len(node.nodes_from) >= 2:
                print(f"Arity 2 node found in graph: {graph}, {node.uid}")
                break


# Check if some of the nodes are kan_linear:
for generation in history.individuals:
    for ind in generation:
        graph = ind.graph
        for node in graph.nodes:
            if node.content["name"] == "kan_linear":
                print(f"KANLinear node found in graph: {graph}, {node.uid}")
                break

# Plot hist of total parameter count:
values = []
for generation in history.individuals:
    for ind in generation:
        values.append(ind.fitness.values[1])

from seaborn import histplot
histplot(values)
plt.show()



# history.show(PlotTypesEnum.operations_animated_bar)
# history.show(PlotTypesEnum.operations_kde)
# history.show(PlotTypesEnum.fitness_line_interactive)

# visualise_pareto(
#     history.final_choices.data, show=True, objectives_names=("LogLoss", "Parameters")
# )
# 
# print(history.get_leaderboard())
# 
# chosen = history.final_choices[0].graph
# PipelineVisualizer(chosen).visualise()

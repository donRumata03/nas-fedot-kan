from nas.composer.nn_composer_requirements import load_default_requirements
from nas.graph.graph_builder.cnn_builder import ConvGraphMaker
from nas.graph.cnn_graph import NasGraph
from nas.graph.graph_builder.base_graph_builder import BaseGraphBuilder


def get_graph() -> NasGraph:
    requirements = load_default_requirements()
    builder = BaseGraphBuilder()
    cnn_builder = ConvGraphMaker(requirements=requirements.model_requirements)
    builder.set_builder(cnn_builder)
    return builder.build()

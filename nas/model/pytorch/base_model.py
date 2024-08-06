from __future__ import annotations

from typing import Union, Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn
import tqdm
from golem.core.dag.graph_node import GraphNode
from numpy import ndarray, dtype
from torch import nn
from torch.utils.data import DataLoader

from nas.graph.base_graph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
from nas.model.model_interface import NeuralSearchModel
from nas.model.pytorch.layers.layer_initializer import TorchLayerFactory

WEIGHTED_NODE_NAMES = ['conv2d', 'linear']


def get_node_input_channels(node: Union[GraphNode, NasNode]):
    n = node.nodes_from[0]
    while not n.parameters.get('out_shape'):
        n = n.nodes_from[0]
    return n.parameters.get('out_shape')


def get_input_shape(node: Union[GraphNode, NasNode],
                    original_shape: Union[Tuple[int], List[int]] = None) -> Optional[Tuple[int, int]]:
    weighted_node = node  # node.nodes_from[0] if node.nodes_from else node
    dim_node = None
    output_channels = None
    side_size = None
    is_start = True
    while weighted_node.name not in WEIGHTED_NODE_NAMES or is_start:
        if not weighted_node.nodes_from:
            break
        else:
            parent_node = weighted_node.nodes_from[0]
        if parent_node.name in ['pooling2d', 'adaptive_pool2d']:
            dim_node = parent_node
        output_channels = parent_node.parameters.get('out_shape')
        is_start = False
        weighted_node = parent_node
    if dim_node:
        # If pooling layer has been found, its output shape/pooling size will be used for side size determination.
        # Otherwise for side calculations conv or linear layer will be used.
        side_size = dim_node.parameters['out_shape'] if dim_node.name == 'adaptive_pool2d' \
            else dim_node.parameters['pool_size']
    elif output_channels:
        # returns side size as 1 for linear layer, for conv layer returns their kernel size
        side_size = weighted_node.parameters.get('kernel_size', 1)
    if node.name == 'flatten':
        output_channels = side_size[0] * output_channels if isinstance(side_size, list) else side_size * output_channels
        side_size = [1, 1]
    return side_size, output_channels


def present_node_dim_info(input_shape, output_shape) -> dict:
    dim_kind = "2d" if len(output_shape) == 3 else "flatten" if len(input_shape) == 3 else "1d"
    res = {
        "dim_kind": dim_kind,
    }

    def present_2d_shape_dim_info(shape) -> dict:
        return {
            "channels": shape[0],
            "side_size": shape[1],
        }

    if dim_kind == "2d":
        res["input"] = present_2d_shape_dim_info(input_shape)
        res["output"] = present_2d_shape_dim_info(output_shape)

    if dim_kind == "flatten":
        res["input"] = present_2d_shape_dim_info(input_shape)
        res["flattened"] = output_shape[0]
        res["output_dim"] = res["flattened"]  # To keep compatibility with 1d

    if dim_kind == "1d":
        res["input_dim"] = input_shape[0]
        res["output_dim"] = output_shape[0]

    # TODO: cache computational complexity here, too

    return res


def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_total_graph_parameters(graph: NasGraph, in_shape, out_shape):
    # First, initialize `NASTorchModel` to cache dims/parameters
    m = NeuralSearchModel(NASTorchModel).compile_model(graph, in_shape, out_shape).model
    # all_layer_parameters = sum(n.content["parameter_count"] for n in graph.nodes)
    # print(f"Counted in full model: {count_parameters(m)}")
    # print(f"Computed by layer: {all_layer_parameters}")

    # Compute sum of all node parameters
    return count_parameters(m)


class NASTorchModel(torch.nn.Module):
    """
    Implementation of Pytorch model class for graph described architectures.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model_layers = None
        self.output_layer = None

        self._graph = None

    def set_device(self, device):
        self.to(device)

    def init_model(self, in_shape: Union[Tuple[int], List[int]], out_shape: int, graph: NasGraph, **kwargs):
        self._graph = graph
        visited_nodes = set()
        visited_node_outputs = {}
        out_shape = out_shape if out_shape > 2 else 1

        def _init_layer(node: Union[GraphNode, NasNode]) -> torch.Tensor:
            if node.nodes_from:
                for n in node.nodes_from:
                    input_tensor = _init_layer(n)
            else:
                input_tensor = torch.rand([2, *in_shape[::-1]])

            if node not in visited_nodes:
                layer_func = TorchLayerFactory.get_layer(node)
                input_channels = input_tensor.shape[1]
                layer = layer_func['weighted_layer'](node, input_dim=input_channels)
                output_tensor = layer(input_tensor)

                # Cache dims data:
                input_shape = input_tensor.shape[1:]
                output_shape = output_tensor.shape[1:]
                node.content["dims"] = present_node_dim_info(input_shape, output_shape)

                # Cache parameter data:
                node.content["parameter_count"] = count_parameters(layer)

                self.__setattr__(f'node_{node.uid}', layer)
                if layer_func.get('normalization'):
                    output_shape = node.parameters['out_shape']
                    normalization_module = layer_func['normalization'](node, input_dim=output_shape)
                    self.__setattr__(f'node_{node.uid}_n', normalization_module)
                    node.content["parameter_count"] += count_parameters(normalization_module)

                if layer_func.get('pooling'):
                    output_shape = node.parameters['out_shape']
                    pooling_module = layer_func['pooling'](node, input_dim=output_shape)
                    self.__setattr__(f'node_{node.uid}_p', pooling_module)
                    node.content["parameter_count"] += count_parameters(pooling_module)

                visited_node_outputs[node.uid] = output_tensor
                visited_nodes.add(node)

            return visited_node_outputs[node.uid]

        _init_layer(graph.root_node)
        self.output_layer = torch.nn.Linear(graph.root_node.content["dims"]["output_dim"], out_shape)

    def forward(self, inputs: torch.Tensor):
        visited_nodes = set()
        node_to_save = dict()

        def _forward_pass_one_layer_recursive(node: Union[GraphNode, NasNode]):
            layer_name = f'node_{node.uid}'
            layer_state_dict = self.__getattr__(layer_name)
            if node in visited_nodes:
                node_to_save[node]['calls'] -= 1
                cond2remove = node_to_save[node]['calls'] == 1
                output = node_to_save[node]['output']
                if cond2remove:
                    del node_to_save[node]
                return output
            first_save_cond = len(self._graph.node_children(node)) > 1 or len(node.nodes_from) > 1
            second_save_cond = node not in node_to_save.keys()
            if first_save_cond and second_save_cond:
                node_to_save[node] = None
            layer_inputs = [_forward_pass_one_layer_recursive(parent) for parent in node.nodes_from] \
                if node.nodes_from else [inputs]
            output = layer_state_dict(layer_inputs[0])
            for supplementary_module in [f'{layer_name}_n', f'{layer_name}_p']:
                if hasattr(self, supplementary_module):
                    output = self.__getattr__(supplementary_module)(output)
            if len(node.nodes_from) > 1:
                shortcut = layer_inputs[-1]
                output += shortcut
            if node.name not in ['pooling2d', 'dropout', 'adaptive_pool2d', 'flatten', 'kan', 'kan_linear']:
                output = TorchLayerFactory.get_activation(node.parameters['activation'])()(output)
            if node in node_to_save.keys():
                node_to_save[node] = {'output': output,
                                      'calls': len(self._graph.node_children(node))}
            visited_nodes.add(node)
            return output

        out = _forward_pass_one_layer_recursive(self._graph.root_node)
        del node_to_save
        return self.output_layer(out)

    def _one_epoch_train(self, train_data: DataLoader, optimizer, loss_fn, device):
        running_loss = 0
        for batch_id, (features_batch, targets_batch) in enumerate(train_data):
            features_batch, targets_batch = features_batch.to(device), targets_batch.to(device)
            optimizer.zero_grad()
            outputs = self.__call__(features_batch)
            loss = loss_fn(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
        running_loss = running_loss / len(train_data)
        return running_loss

    def validate(self, val_data: DataLoader, loss_fn, device, **kwargs) -> Dict:
        metrics_to_calc = kwargs.get('metrics')
        metrics = {'val_loss': 0}
        for batch_id, (features_batch, targets_batch) in enumerate(val_data):
            features_batch, targets_batch = features_batch.to(device), targets_batch.to(device)
            outputs = self.__call__(features_batch)
            loss = loss_fn(outputs, targets_batch)
            metrics['val_loss'] += loss.detach().cpu().item()
            if metrics_to_calc:
                for metric_name, metric_func in metrics_to_calc.items():
                    if metrics.get(f'val_{metric_name}') is None:
                        metrics[f'val_{metric_name}'] = 0
                    metrics[f'val_{metric_name}'] += metric_func(outputs.detach().cpu().numpy(),
                                                                 targets_batch.detach().cpu().numpy())

        metrics = {key: val / len(val_data) for key, val in metrics.items()}
        return metrics

    def fit(self, train_data: DataLoader,
            loss,
            val_data: Optional[DataLoader] = None,
            optimizer=torch.optim.AdamW,
            epochs: int = 1,
            device: str = 'cpu',
            **kwargs):
        """
        This function trains the pytorch module using given parameters
        """
        self.set_device(device)
        metrics_to_val = kwargs.get('metrics')
        metrics = dict()
        optim = optimizer(self.parameters(), lr=kwargs.get('lr', 1e-3))
        # pbar = tqdm.trange(epochs, desc='Fitting graph', leave=False, position=0)
        for epoch in range(epochs):
            self.train(mode=True)
            train_loss = self._one_epoch_train(train_data, optim, loss, device)
            metrics['train_loss'] = train_loss
            if val_data:
                with torch.no_grad():
                    self.eval()
                    val_metrics = self.validate(val_data, loss, device, metrics=metrics_to_val)
                metrics = metrics | val_metrics

    def predict(self,
                test_data: DataLoader,
                device: str = 'cpu') -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        This method implements prediction on data loader.
        Returns tuple of predicted probabilities and target labels.
        """
        self.set_device(device)
        self.eval()
        results = []
        targets_lst = []
        activation = torch.nn.Softmax(-1)
        with torch.no_grad():
            for features, targets in test_data:
                features, targets = features.to(device), targets.to(device)
                predictions = self.__call__(features)
                results.extend(activation(predictions).detach().cpu().tolist())
                targets_lst.extend(torch.argmax(targets, dim=-1).detach().cpu().tolist())
        return np.array(results), np.array(targets_lst)

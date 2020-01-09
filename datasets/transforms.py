import numpy as np
import re
import torch
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from functools import partial


class MeshToNormal(object):
    """ Computes mesh normals (IN PROGRESS)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        if hasattr(data, "face"):
            pos = data.pos
            face = data.face
            vertices = [pos[f] for f in face]
            normals = torch.cross(
                vertices[0] - vertices[1], vertices[0] - vertices[2], dim=1
            )
            normals = F.normalize(normals)
            data.normals = normals
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class MultiScaleTransform(object):
    """ Pre-computes a sequence of downsampling / neighboorhood search
    """

    def __init__(self, strategies, precompute_multi_scale=False):
        self.strategies = strategies
        self.precompute_multi_scale = precompute_multi_scale
        if self.precompute_multi_scale and not bool(strategies):
            raise Exception(
                "Strategies are empty and precompute_multi_scale is set to True"
            )
        self.num_layers = len(self.strategies.keys())

    @staticmethod
    def __inc__wrapper(func, special_params):
        def new__inc__(key, num_nodes, special_params=None, func=None):
            if key in special_params:
                return special_params[key]
            else:
                return func(key, num_nodes)

        return partial(new__inc__, special_params=special_params, func=func)

    def __call__(self, data: Data):
        if self.precompute_multi_scale:
            # Compute sequentially multi_scale indexes on cpu
            special_params = {}
            pos = data.pos
            batch = torch.zeros((pos.shape[0],), dtype=torch.long)
            for index in range(self.num_layers):
                sampler, neighbour_finder = self.strategies[index]
                idx = sampler(pos, batch)
                row, col = neighbour_finder(pos, pos[idx], batch, batch[idx])
                edge_index = torch.stack([col, row], dim=0)

                index_name = "index_{}".format(index)
                edge_name = "edge_index_{}".format(index)

                setattr(data, index_name, idx)
                setattr(data, edge_name, edge_index)

                num_nodes_for_edge_index = torch.from_numpy(
                    np.array([pos.shape[0], pos[idx].shape[0]])
                ).unsqueeze(-1)

                special_params[index_name] = num_nodes_for_edge_index[0]

                special_params[edge_name] = num_nodes_for_edge_index
                pos = pos[idx]
                batch = batch[idx]
            func_ = self.__inc__wrapper(data.__inc__, special_params)
            setattr(data, "__inc__", func_)
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

from torch_geometric.nn import PointConv
from models.core_modules import *
from models.core_sampling_and_search import MultiscaleRadiusNeighbourFinder, FPSSampler


class SAModule(BaseMSConvolutionDown):
    def __init__(self, ratio=None, radius=None, radius_num_point=None, down_conv_nn=None, nb_feature=None, *args, **kwargs):
        super(SAModule, self).__init__(FPSSampler(ratio=ratio),
                                       MultiscaleRadiusNeighbourFinder(radius, max_num_neighbors=radius_num_point), *args, **kwargs)

        local_nn = MLP(down_conv_nn) if down_conv_nn is not None else None

        self._conv = PointConv(local_nn=local_nn, global_nn=None)
        self._radius = radius
        self._ratio = ratio
        self._num_points = radius_num_point

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)

    def extra_repr(self):
        return '{}(ratio {}, radius {}, radius_points {})'.format(self.__class__.__name__, self._ratio, self._radius, self._num_points)

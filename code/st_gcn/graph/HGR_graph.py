import numpy as np
from . import tools
# Edge format: (origin, neighbor)
num_node = 22
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8),
                                  (8, 9), (1, 10), (10, 11), (11, 12), (12, 13), (1, 14), (14, 15),
                                  (15, 16), (16, 17), (1, 18), (18, 19), (19, 20), (20, 21)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A
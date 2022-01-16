#!/usr/bin/env python3
from ..common import Nodes, Edges
from typing import List, Sequence, Tuple


class EgoGraph(object):

    """
    The abstraction of a mini-batch of ego-graphs.

    :param centre_nodes: :class:`graph2tensor.common.Nodes` object
    """

    def __init__(self, centre_nodes: Nodes):
        self._centre_nodes = centre_nodes
        self._paths = []

    def add_path(self, path: Sequence[Tuple[Edges, Nodes]]):
        """
        Add a new path into the ego-graph

        :param path: list of (:class:`graph2tensor.common.Edges`,
                     :class:`graph2tensor.common.Nodes`) tuple represent a meta-path,
                     each path added into the sub-graph should have same length
        """
        self._paths.append(path)

    @property
    def centre_nodes(self) -> Nodes:
        """
        Get the centre nodes of the ego-graphs.

        :return: the centre nodes of the ego-graphs
        """
        return self._centre_nodes

    @property
    def paths(self) -> List[List[Tuple[Edges, Nodes]]]:
        """
        Get all the paths of the ego-graphs.

        :return: all the paths of the ego-graphs
        """
        return self._paths

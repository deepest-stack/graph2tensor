#!/usr/bin/env python3

import regex
from .base_sampler import BaseSampler
from ..egograph import EgoGraph
from ..common import Nodes
from typing import Sequence, Union
import numpy as np


class MetaPathSampler(BaseSampler):
    """
    An ego-graph sampler that samples ego-graph along with the paths specified.

    :param graph: graph to sample from
    :param meta_paths: list of string which specifies sampling paths and
                        each string in the list defined a meta-path.
                        The pattern of the string should be something like:
                        "(N1) -[E1]- (N2) -[E2]- (N2) -[E3]- (N3)".
                        It will define a path with 3 hops, along with edge types: E1 -> E2 -> E3.
                        Each path defined in `meta_paths` should have same length, the src and dst node types
                        of edges should match with graph schema, and the type of initial node of each path,
                        i.e., the type of the centre nodes, should be consistent.
                        To be a valid path, edge should have common node type with the adjacent edge in the path.
    :param expand_factors: `list` or `int`, the number of neighbors to sample for each node at each hop.
                            If pass an int, all hops in all paths use the same expand factor, if pass a list,
                            it should match with the meta paths, i.e., same number of paths and same length of paths.
    :param strategies: `list` or `str`, the sampling strategy at each hop.
                        If pass a string, all hops in all paths use the same strategy, if pass a list,
                        it should match with the meta paths, i.e., same number of paths and same length of paths.
    """

    def __init__(self, graph, meta_paths, expand_factors, strategies):
        self._graph = graph
        self._centre_node_type = None
        self._meta_paths = self._parse_paths(meta_paths)
        if isinstance(expand_factors, list):
            self._expand_factors = expand_factors
        elif isinstance(expand_factors, int):
            path_num, path_len = len(self._meta_paths), len(self._meta_paths[0])
            self._expand_factors = [[expand_factors for _ in range(path_len)] for _ in range(path_num)]
        else:
            raise ValueError("`expand_factors` expected `list` or `int`, but received `%s`"
                             % (expand_factors.__class__.__name__, ))
        if isinstance(strategies, list):
            self._strategies = strategies
        elif isinstance(strategies, str):
            path_num, path_len = len(self._meta_paths), len(self._meta_paths[0])
            self._strategies = [[strategies for _ in range(path_len)] for _ in range(path_num)]
        else:
            raise ValueError("`strategies` expected `list` or `str`, but received `%s`"
                             % (strategies.__class__.__name__, ))

    def _parse_paths(self, meta_paths):
        pt = regex.compile("\((\S+?)\)\s*-\[(\S+?)\]-\s*\((\S+?)\)")
        paths = []
        path_length = None
        for path in meta_paths:
            edges = []
            hops = pt.findall(path, overlapped=True)
            if not hops:
                raise ValueError("Invalid path definition: %s" % (path, ))
            if self._centre_node_type is None: self._centre_node_type = hops[0][0]
            if hops[0][0] != self._centre_node_type:
                raise ValueError("Centre node type in each path should be consistent")
            if path_length is None: path_length = len(hops)
            if len(hops) != path_length:
                raise ValueError("Each path defined in meta_paths should have same length")
            for left_type, edge_type, right_type in hops:
                if edge_type not in self._graph.schema["edges"]:
                    raise ValueError("graph has no edge type: %s" % edge_type)
                src_type, dst_type = self._graph.get_src_type(edge_type), self._graph.get_dst_type(edge_type)
                left_type_ref, right_type_ref = (src_type, dst_type)
                if left_type_ref != left_type or right_type_ref != right_type:
                    raise ValueError("The src/dst node types of edge `%s` in path %s not match with graph schema"
                                     % (edge_type, path))
                edges.append(edge_type)
            paths.append(edges)
        return paths

    def sample(self, ids: Union[np.ndarray, Sequence[int]], **kwargs) -> EgoGraph:
        """
        Sample ego-graphs start from ids

        :param ids: the ids of the start nodes
        :param kwargs: see :func:`graph2tensor.client.NumpyGraph.sample_neighbors` for details
        :return: tuple whose element is 1-d `np.ndarray` which representing a walk path
        """
        ego_graph = EgoGraph(Nodes(self._centre_node_type, ids))
        for path, expand_factors, strategies in \
                zip(self._meta_paths, self._expand_factors, self._strategies):
            hops = []
            ids = ego_graph.centre_nodes.ids
            for edge_type, expand_factor, strategy in \
                    zip(path, expand_factors, strategies):
                dst_nodes, edges = self._graph.sample_neighbors(
                    ids,
                    edge_type,
                    num=expand_factor,
                    strategy=strategy,
                    **kwargs
                )
                ids = dst_nodes.ids
                hops.append((edges, dst_nodes))
            ego_graph.add_path(hops)
        return ego_graph

#!/usr/bin/env python3
from typing import Union, Sequence, Tuple, List
import numpy as np
from .base_sampler import BaseSampler
from .meta_path_sampler import MetaPathSampler


class RandomWalker(BaseSampler):
    r"""
    A walker that perform random walk.

    :param graph: the graph to sample from
    :param edge_type: the type of edge to walk along
    :param walk_length: the length to walk
    :param use_edge_probs: whether use edge probabilities as neighbour's distribution,
                           if not or edge probabilities is not found, uniform
                           distribution will be applied
    :param discard_frequent_nodes: whether or not discard frequent nodes
    :param freq_th: the frequency threshold for frequent nodes
    """

    def __init__(self, graph, edge_type, walk_length=10,
                 use_edge_probs=False, discard_frequent_nodes=True, freq_th=1e-5):
        self._graph = graph
        self._edge_type = edge_type
        self._walk_length = walk_length
        self._use_edge_probs = use_edge_probs
        self._discard_frequent_nodes = discard_frequent_nodes
        self._freq_th = freq_th

    def sample(self, ids: Union[np.ndarray, Sequence[int]], **kwargs) -> Tuple[List[int], ...]:
        """
        Perform random walk starting from ids.

        :param ids: the id of nodes to start walk from
        :return: tuple whose element is list which representing a walk path
        """
        return self._graph.random_walk(
            ids, self._edge_type, self._walk_length,
            use_edge_probs=self._use_edge_probs,
            discard_frequent_nodes=self._discard_frequent_nodes,
            freq_th=self._freq_th
        )


class Node2VecWalker(BaseSampler):
    r"""
    A walker that perform walk following the strategy described in
    `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__.

    :param graph: the graph to sample from
    :param edge_type: the type of edge to walk along
    :param walk_length: the length to walk
    :param p: return parameter, see `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__
              for more details
    :param q: in-out parameter, see `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__
              for more details
    :param use_edge_probs: whether use edge probabilities as neighbour's distribution,
                           if not or edge probabilities is not found, uniform
                           distribution will be applied
    :param discard_frequent_nodes: whether or not discard frequent nodes
    :param freq_th: the frequency threshold for frequent nodes
    """

    def __init__(self, graph, edge_type, walk_length, p=1.0, q=1.0,
                 use_edge_probs=False, discard_frequent_nodes=True, freq_th=1e-5):
        self._graph = graph
        self._edge_type = edge_type
        self._walk_length = walk_length
        self._p = p
        self._q = q
        self._use_edge_probs = use_edge_probs
        self._discard_frequent_nodes = discard_frequent_nodes
        self._freq_th = freq_th

    def sample(self, ids: Union[np.ndarray, Sequence[int]], **kwargs) -> Tuple[List[int], ...]:
        """
        Perform node2vec walk starting from ids.

        :param ids: the id of nodes to start walk from
        :return: tuple whose element is list which representing a walk path
        """
        return self._graph.node2vec_walk(
            ids, self._edge_type, self._walk_length, self._p, self._q,
            use_edge_probs=self._use_edge_probs,
            discard_frequent_nodes=self._discard_frequent_nodes,
            freq_th=self._freq_th
        )


class MetaPathRandomWalker(MetaPathSampler):

    def __init__(self, graph, meta_path, walk_length):
        self._walk_length = walk_length
        super(MetaPathRandomWalker, self).__init__(
            graph, meta_paths=[meta_path, ],
            expand_factors=1, strategies='random', include_edge=False
        )

    def _parse_paths(self, meta_paths):
        paths = super(MetaPathRandomWalker, self)._parse_paths(meta_paths)
        path = []
        for i in range(self._walk_length):
            path.append(paths[0][i % len(paths[0])])
        return [path, ]

    def sample(self, ids: Union[np.ndarray, Sequence[int]], **kwargs):

        def _ego2paths():
            path = ego_graph.paths[0]
            batch_size, path_length = len(ids), len(path)
            paths = np.zeros((batch_size, path_length+1), dtype=np.int64) - 1
            paths[:, 0] = ego_graph.centre_nodes.ids
            idx = np.arange(batch_size)
            for i in range(path_length):
                idx = idx[np.where(path[i][1].offset != 0)[0]]
                paths[idx, i+1] = path[i][1].ids
            r, c = np.where(paths == -1)
            path_tail_indices = np.ones(batch_size, dtype=np.int32) * (path_length+1)
            v, i = np.unique(r, return_index=True)
            path_tail_indices[v] = c[i]
            return tuple([paths[i][:path_tail_indices[i]] for i in range(batch_size)])

        ego_graph = super(MetaPathRandomWalker, self).sample(
            ids, **kwargs
        )
        return _ego2paths()


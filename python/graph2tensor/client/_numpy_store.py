#!/usr/bin/env python3
import numpy as np
from .sampler import sample_topk_neighbors, sample_neighbors_randomly
from .sampler import random_walk, node2vec_walk
from scipy.sparse import csr_matrix
from collections import Counter


class _NumpyAttributeStore(object):

    def __init__(self):
        self._node_attrs = {}
        self._edge_attrs = {}
        self._node_label = {}

    def add_node(self, node_type, node_attrs, node_label):
        if node_attrs:
            self._node_attrs[node_type] = {}
            for attr_name, attr_data in node_attrs.items():
                if attr_name in self._node_attrs[node_type]:
                    raise Warning("Multiple assignments to attribute %s of node %s, "
                                  "only the last take effect" % (attr_name, node_type))
                if attr_data.shape.__len__() == 1:
                    attr_data = attr_data.reshape(-1, 1)
                self._node_attrs[node_type][attr_name] = np.concatenate(
                    (np.zeros_like(attr_data[0:1], dtype=attr_data.dtype), attr_data),
                    axis=0
                )
        if node_label is not None:
            self._node_label[node_type] = np.concatenate(
                    (np.zeros_like(node_label[0:1], dtype=node_label.dtype), node_label),
                    axis=0
                )

    def add_edge(self, edge_type, directed, edge_attrs):
        if edge_attrs:
            self._edge_attrs[edge_type] = {}
            for attr_name, attr_data in edge_attrs.items():
                if attr_name in self._edge_attrs[edge_type]:
                    raise Warning("Multiple assignments to attribute %s of edge %s, "
                                  "only the last take effect" % (attr_name, edge_type))
                if attr_data.shape.__len__() == 1:
                    attr_data = attr_data.reshape(-1, 1)
                self._edge_attrs[edge_type][attr_name] = np.concatenate(
                    (np.zeros_like(attr_data[0:1], dtype=attr_data.dtype), attr_data),
                    axis=0
                )

    def lookup_nodes(self, nodes, columns, graph_schema, **kwargs):
        node_attrs = {}
        for col in columns:
            node_attrs[col] = np.take(self._node_attrs[nodes.node_type][col], nodes.ids, axis=0)
        return node_attrs

    def lookup_edges(self, edges, columns, graph_schema, **kwargs):
        edge_attrs = {}
        for col in columns:
            edge_attrs[col] = np.take(self._edge_attrs[edges.edge_type][col], edges.edge_ids, axis=0)
        return edge_attrs

    def get_node_attr_data(self, node_type, attr_name):
        return self._node_attrs[node_type][attr_name]

    def get_edge_attr_data(self, edge_type, attr_name):
        return self._edge_attrs[edge_type][attr_name]

    def get_node_label(self, node_type):
        return self._node_label[node_type]

    def to_config(self):
        return {
            "node_attrs": self._node_attrs,
            "edge_attrs": self._edge_attrs,
            "node_label": self._node_label
        }

    @classmethod
    def from_config(cls, config):
        g = cls()
        g._node_attrs = config["node_attrs"]
        g._edge_attrs = config["edge_attrs"]
        g._node_label = config["node_label"]
        return g


class _NumpyRelationStore(object):

    def __init__(self):
        self._edges = dict()

    def add_edge(self, edge_type, directed, src_ids, dst_ids, edge_ids=None, edge_probs=None):
        src_ids, dst_ids = np.asarray(src_ids, dtype=np.int64).ravel(), \
                           np.asarray(dst_ids, dtype=np.int64).ravel()
        if src_ids.shape != dst_ids.shape:
            raise ValueError("`src_ids` and `dst_ids` should have same size")
        if src_ids.min() == 0 or dst_ids.min() == 0:
            raise ValueError("node ids can not be 0")
        if edge_probs is not None:
            edge_probs = np.asarray(edge_probs, dtype=np.float64).ravel()
            if edge_probs.shape != src_ids.shape:
                raise ValueError("`edge_probs` should have same size as `src_ids`")
        if edge_ids is not None:
            edge_ids = np.asarray(edge_ids, dtype=np.int64).ravel()
            if edge_ids.shape != src_ids.shape:
                raise ValueError("`edge_ids` should have same size as `src_ids`")
            if edge_ids.min() == 0:
                raise ValueError("edge ids can not be 0")
        if np.unique(
                np.concatenate((src_ids, dst_ids)).reshape((2, -1)),
                axis=1).shape[1] < src_ids.shape[0]:
            raise ValueError("multiple edges are not allowed")
        if not directed and np.unique(
                np.concatenate((src_ids, dst_ids, dst_ids, src_ids)).reshape((2, -1)),
                axis=1).shape[1] < src_ids.shape[0] * 2:
            raise ValueError("bi-direction edges or self-loop are not allowed in undirected edges")
        # edge ids start from 1 rather than 0
        data = np.arange(1, src_ids.shape[0]+1, dtype=np.int64)
        if directed:
            adj_mat = csr_matrix((data, (src_ids, dst_ids)),)
        else:
            adj_mat = csr_matrix(
                (np.concatenate((data, data)),
                 (np.concatenate((src_ids, dst_ids)), np.concatenate((dst_ids, src_ids))))
            )
        # for undirected edge, adj_mat.data has been doubled,
        # thus, edge_ids & edge_probs will also be doubled
        if edge_ids is not None:
            edge_ids = edge_ids[adj_mat.data-1]
        else:
            edge_ids = adj_mat.data
        if edge_probs is not None:
            edge_probs = edge_probs[adj_mat.data-1]

        self._edges[edge_type] = {
            "adj_mat": adj_mat,
            "edge_ids": edge_ids,
            "edge_probs": edge_probs
        }

    def get_edge_ids(self, edge_type):
        src_ids, dst_ids = self._edges[edge_type]['adj_mat'].nonzero()
        return src_ids, dst_ids, self._edges[edge_type]['edge_ids']

    def get_edge_probs(self, edge_type):
        return self._edges[edge_type]['edge_probs']

    def sample_all_neighbors(self, ids, edge_type, **kwargs):
        sub_adj = self._edges[edge_type]["adj_mat"][ids]
        nbr_ids = sub_adj.indices
        offset = sub_adj.indptr[1:] - sub_adj.indptr[:-1]
        edge_ids = sub_adj.data
        # search nodes that have no neighbour
        idx = np.where(offset == 0)[0]
        # insert sentinel node & edge ids
        # and update offset
        x = np.cumsum(offset)
        nbr_ids = np.insert(nbr_ids, x[idx], 0)
        edge_ids = np.insert(edge_ids, x[idx], 0)
        offset[idx] = 1
        return nbr_ids, offset, edge_ids

    def sample_topk_neighbors(self, ids, edge_type, k, **kwargs):
        if self._edges[edge_type]["edge_probs"] is None:
            raise ValueError("`edge_probs` is needed for topK sampling")
        args = {
            "k": k,
            "ids0": ids,
            "ids1": ids+1,
            "nbr_ids": self._edges[edge_type]["adj_mat"].indices,
            "nbr_ptrs": self._edges[edge_type]["adj_mat"].indptr,
            "edge_probs": self._edges[edge_type]["edge_probs"]
        }
        num_threads = kwargs.get("num_threads", -1)
        if num_threads > 0:
            args['num_threads'] = num_threads
        args["edge_ids"] = self._edges[edge_type]["edge_ids"]
        rets = sample_topk_neighbors(**args)
        nbr_ids = np.concatenate(rets[:ids.shape[0]])
        offset = rets[-1]
        edge_ids = np.concatenate(rets[ids.shape[0]:-1])
        return nbr_ids, offset, edge_ids

    def sample_neighbors_randomly(self, ids, edge_type, n, **kwargs):
        args = {
            "n": n,
            "replace": kwargs.get("replace", False),
            "ids0": ids,
            "ids1": ids+1,
            "nbr_ids": self._edges[edge_type]["adj_mat"].indices,
            "nbr_ptrs": self._edges[edge_type]["adj_mat"].indptr,
        }
        if kwargs.get("use_edge_probs", False) and self._edges[edge_type]["edge_probs"] is not None:
            args["edge_probs"] = self._edges[edge_type]["edge_probs"]
        num_threads = kwargs.get("num_threads", -1)
        if num_threads > 0:
            args['num_threads'] = num_threads
        args["edge_ids"] = self._edges[edge_type]["edge_ids"]
        rets = sample_neighbors_randomly(**args)
        nbr_ids = np.concatenate(rets[:ids.shape[0]])
        offset = rets[-1]
        edge_ids = np.concatenate(rets[ids.shape[0]:-1])
        return nbr_ids, offset, edge_ids

    def to_config(self):
        return self._edges

    @classmethod
    def from_config(cls, config):
        obj = cls()
        obj._edges = config
        return obj

    def _get_nbr_freqs(self, edge_type):
        if "nbr_freqs" not in self._edges[edge_type]:
            freq = Counter()
            src_ids, dst_ids, _ = self.get_edge_ids(edge_type)
            freq.update(src_ids)
            freq.update(dst_ids)
            nbr_freqs = np.zeros(max(freq.keys()) + 1)
            nbr_freqs[list(freq.keys())] = list(freq.values())
            nbr_freqs = nbr_freqs / nbr_freqs.sum()
            self._edges[edge_type]['nbr_freqs'] = nbr_freqs[dst_ids]
        return self._edges[edge_type]['nbr_freqs']

    def random_walk(self, ids, edge_type, walk_length, **kwargs):
        args = {
            "walk_length": walk_length,
            "ids": ids.astype(np.int64),
            "nbr_ids": self._edges[edge_type]["adj_mat"].indices,
            "nbr_ptrs": self._edges[edge_type]["adj_mat"].indptr,
        }
        if kwargs.get("use_edge_probs", False) and self._edges[edge_type]["edge_probs"] is not None:
            args["edge_probs"] = self._edges[edge_type]["edge_probs"]
        if kwargs.get("discard_frequent_nodes", False):
            args["nbr_freqs"] = self._get_nbr_freqs(edge_type)
            args["freq_th"] = kwargs.get("freq_th", 1e-5)
        return random_walk(**args)

    def node2vec_walk(self, ids, edge_type, walk_length, p, q, **kwargs):
        args = {
            "walk_length": walk_length,
            "ids": ids.astype(np.int64),
            "nbr_ids": self._edges[edge_type]["adj_mat"].indices,
            "nbr_ptrs": self._edges[edge_type]["adj_mat"].indptr,
            "p": float(p),
            "q": float(q)
        }
        if kwargs.get("use_edge_probs", False) and self._edges[edge_type]["edge_probs"] is not None:
            args["edge_probs"] = self._edges[edge_type]["edge_probs"]
        if kwargs.get("discard_frequent_nodes", False):
            args["nbr_freqs"] = self._get_nbr_freqs(edge_type)
            args["freq_th"] = kwargs.get("freq_th", 1e-5)
        return node2vec_walk(**args)

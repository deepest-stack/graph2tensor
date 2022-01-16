#!/usr/bin/env python3
from .base_graph import BaseGraph
from ._numpy_store import _NumpyAttributeStore, _NumpyRelationStore
from ..common import Nodes, Edges
import numpy as np


class NumpyGraph(BaseGraph):
    """
    An object for graph data manipulating.
    """

    def __init__(self):
        super(NumpyGraph, self).__init__()
        self._relation_store = _NumpyRelationStore()
        self._attribute_store = _NumpyAttributeStore()

    def lookup_edges(self, edges, **kwargs):
        columns = [attr for attr in self.schema["edges"][edges.edge_type]["attrs"]]
        return self._attribute_store.lookup_edges(
            edges,
            columns,
            self.schema,
            **kwargs
        )

    def lookup_nodes(self, nodes, **kwargs):
        columns = [attr for attr in self.schema["nodes"][nodes.node_type]["attrs"]]
        return self._attribute_store.lookup_nodes(nodes, columns, self.schema, **kwargs)

    def add_node(self, node_type, attrs_info, labeled, **kwargs):
        """
        Add a new type of node into graph

        :param node_type: the type of node
        :param attrs_info: name & type of attributes of node, `dict` or `list` or `tuple`.
                            For `dict`, the key should be attribute name and value be type.
                            For `list` or `tuple`, each element should be a key-value pair,
                            e.g. (attr_name, attr_type) or [attr_name, attr_type].
                            In any situation above, the attribute type can be one of `int`,
                            `float` or `str`, and for array, use `int[N]`, `float[N]` or `str[N]`
                            in which `N` is the size of array,
                            otherwise, a `ValueError` exception will be raised.
        :param labeled: whether node is labeled
        :param kwargs: other valid args: `node_label`, the data of node label in `np.ndarray`
                       format when node is labeled; `node_attrs`, a `dict` whose keys are
                       attribute names and values are attribute data in `np.ndarray` format
        """
        super(NumpyGraph, self).add_node(node_type, attrs_info, labeled)
        node_label = kwargs.get("node_label")
        if labeled and node_label is None:
            raise ValueError("`node_label` was missed for labeled node: %s" % node_type)
        self._attribute_store.add_node(node_type, kwargs.get("node_attrs"), node_label)

    def add_edge(self, edge_type, src_type, dst_type, attrs_info, directed=True, **kwargs):
        """
        Add a new type of edge into graph

        :param edge_type: the type of edge
        :param src_type: the type of source node
        :param dst_type: the type of destination node
        :param attrs_info: name & type of attributes of edges, `dict` or `list` or `tuple`.
                            For `dict`, the key should be attribute name and value be type.
                            For `list` or `tuple`, each element should be a key-value pair,
                            e.g. (attr_name, attr_type) or [attr_name, attr_type].
                            In any situation above, the attribute type can only be one of `int`,
                            `float` or `str`, and for array, use `int[N]`, `float[N]` or `str[N]`
                            in which `N` is the size of array, otherwise, a `ValueError`
                            exception will be raised.
        :param directed: whether edge is directed, if not, an reversed edge will also be added,
                        for undirected edge, the `src_type` and `dst_type` must be same, i.e.,
                        the type of source node and destination node of undirected edge should
                        be same. default to `True`
        :param kwargs: other valid args: `src_ids`, required, the ids of source nodes in `np.ndarray`
                       format; `dst_ids`, required, the ids of destination nodes in `np.ndarray`
                       format; `edge_ids`, optional, the ids of edges; `edge_probs`, the probabilities
                       of edges, when sample neighbours along the edges using "topk" strategy,
                       `edge_probs` is required and will be used to computed the topk neighbours,
                       for "random" sampling strategy, `edge_probs` is optional, if specified, neighbours
                       will be sampled based on edge probabilities, otherwise sampled uniformly;
                       `edge_attrs`, a `dict` whose keys are attribute names and values are attribute
                       data in `np.ndarray` format
        """
        super(NumpyGraph, self).add_edge(edge_type, src_type, dst_type, attrs_info, directed)
        self._relation_store.add_edge(edge_type, directed,
                                      kwargs["src_ids"],
                                      kwargs["dst_ids"],
                                      kwargs.get("edge_ids"),
                                      kwargs.get("edge_probs"))
        self._attribute_store.add_edge(edge_type, directed, kwargs.get("edge_attrs"))

    def get_edge_attr_data(self, edge_type, attr_name, **kwargs):
        if edge_type not in self.schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edge_type)
        if attr_name not in self.get_edge_attr_info(edge_type):
            raise ValueError("edge %s has no attribute %s" % (edge_type, attr_name))
        return self._attribute_store.get_edge_attr_data(edge_type, attr_name)

    def get_edge_ids(self, edge_type, **kwargs):
        """
        Get src_ids, dst_ids and edge_ids of all the edges with specified type.

        :param edge_type: edge type
        :return: src_ids, dst_ids and edge_ids, all in `np.ndarray` format
        """
        if edge_type not in self.schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edge_type)
        return self._relation_store.get_edge_ids(edge_type)

    def get_edge_probs(self, edge_type, **kwargs):
        """
        Get the edge probabilities of all the edges with specified type.

        :param edge_type: edge type
        :return: the edge probabilities in `np.ndarray` format
        """
        if edge_type not in self.schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edge_type)
        return self._relation_store.get_edge_probs(edge_type)

    def get_node_attr_data(self, node_type, attr_name, **kwargs):
        if node_type not in self.schema["nodes"]:
            raise ValueError("graph has no node type: %s" % node_type)
        if attr_name not in self.get_node_attr_info(node_type):
            raise ValueError("node %s has no attribute %s" % (node_type, attr_name))
        return self._attribute_store.get_node_attr_data(node_type, attr_name)

    def get_node_label(self, node_type, **kwargs):
        if node_type not in self.schema["nodes"]:
            raise ValueError("graph has no node type: %s" % node_type)
        if not self.is_node_labeled(node_type):
            raise ValueError("node %s was not labeled" % (node_type,))
        return self._attribute_store.get_node_label(node_type)

    def to_config(self):
        return {
            "rel": self._relation_store.to_config(),
            "attr": self._attribute_store.to_config(),
            "schema": self.schema
        }

    @classmethod
    def from_config(cls, config):
        rel = _NumpyRelationStore.from_config(config["rel"])
        attr = _NumpyAttributeStore.from_config(config["attr"])
        obj = cls()
        obj._relation_store = rel
        obj._attribute_store = attr
        obj._schema = config["schema"]
        return obj

    def sample_neighbors(self, ids, edge_type, num=None, strategy="random", **kwargs):
        """
        Sample neighbors along with the specified edge according to the specified strategy.

        :param ids: the id of src nodes
        :param edge_type: the type of the edge to sample along with
        :param num: the number of neighbors to sample for each node
        :param strategy: sampling strategy, several strategies are supported and default as "random".
                        "random": sample `num` neighbors randomly with/without replacement,
                        "all": sample all neighbors and ignore the `num` param,
                        "topk": sample `num` neighbors with highest edge weights,
                        return all neighbors if not enough.
        :param kwargs: other valid args: `num_threads`, specify the number of working threads;
                       `replace`, whether the sample is with or without replacement, only valid
                       for "random" sampling strategy; `use_edge_probs`, whether use edge probabilities
                       as neighbour's distribution, if not or edge probabilities is not found, uniform
                       distribution will be applied, only valid for "random" sampling strategy.
        :return: A :class:`graph2tensor.common.Nodes` object for neighbors,
                 an :class:`graph2tensor.common.Edges` object or `None` for edges.
        """
        dst_type = self.get_dst_type(edge_type)
        if ids.shape[0] == 0:
            dst_nodes = Nodes(dst_type, np.array([], dtype=np.int64), np.array([], dtype=np.int32))
            edges = Edges(
                edge_type,
                edge_ids=np.array([], dtype=np.int64),
                src_ids=np.array([], dtype=np.int64),
                dst_ids=np.array([], dtype=np.int64)
            )
            return dst_nodes, edges
        if strategy.upper() == "ALL":
            nbr_ids, offset, edge_ids = self._relation_store.sample_all_neighbors(
                ids, edge_type, **kwargs)
        elif strategy.upper() == "TOPK":
            nbr_ids, offset, edge_ids = self._relation_store.sample_topk_neighbors(
                ids, edge_type, num, **kwargs)
        else:
            nbr_ids, offset, edge_ids = self._relation_store.sample_neighbors_randomly(
                ids, edge_type, num, **kwargs)
        edges = Edges(edge_type, edge_ids=edge_ids, src_ids=np.repeat(ids.ravel(), offset), dst_ids=nbr_ids)
        return Nodes(dst_type, nbr_ids, offset), edges

    def random_walk(self, ids: np.ndarray, edge_type: str,
                    walk_length: int, **kwargs):
        """
        Walk `walk_length` steps from `ids` along with edge `edge_type` randomly.

        :param ids: the id of nodes to start walk from
        :param edge_type: the type of edge to walk along
        :param walk_length: the length to walk
        :param kwargs: other valid args: `use_edge_probs`, whether use edge probabilities
                       as neighbour's distribution, if not or edge probabilities is not found, uniform
                       distribution will be applied; `discard_frequent_nodes`, whether or not discard
                       frequent nodes; `freq_th`, the frequency threshold for frequent nodes.
        :return: tuple whose element is list which representing a walk path
        """
        return self._relation_store.random_walk(ids, edge_type, walk_length, **kwargs)

    def node2vec_walk(self, ids: np.ndarray, edge_type: str,
                      walk_length: int, p: float, q: float, **kwargs):
        """
        Walk `walk_length` steps from `ids` along with edge `edge_type` following
        the way described in `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__.

        :param ids: the id of nodes to start walk from
        :param edge_type: the type of edge to walk along
        :param walk_length: the length to walk
        :param p: return parameter, see `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__
                  for more details
        :param q: in-out parameter, see `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__
                  for more details
        :param kwargs: other valid args: `use_edge_probs`, whether use edge probabilities
                       as neighbour's distribution, if not or edge probabilities is not found, uniform
                       distribution will be applied; `discard_frequent_nodes`, whether or not discard
                       frequent nodes; `freq_th`, the frequency threshold for frequent nodes.
        :return: tuple whose element is list which representing a walk path
        """
        return self._relation_store.node2vec_walk(ids, edge_type, walk_length, p, q, **kwargs)


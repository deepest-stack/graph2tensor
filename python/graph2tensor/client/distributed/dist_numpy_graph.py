#!/usr/bin/env python3
from ..base_graph import BaseGraph
from ...common import Nodes, Edges
import numpy as np
import grpc
import json
from dist_numpy_attribute_store_pb2 import AttrRequest
from dist_numpy_relation_store_pb2 import BatchingProto, SamplingRequest, WalkRequest
from ndarray_pb2 import NdarrayProto
from dist_numpy_attribute_store_pb2_grpc import DistNumpyAttributeStoreStub
from dist_numpy_relation_store_pb2_grpc import DistNumpyRelationStoreStub


class DistNumpyGraph(BaseGraph):
    """
    The client for distributed graph data manipulating.
    """

    def __init__(self):
        super(DistNumpyGraph, self).__init__()
        self._node_stubs = {}
        self._edge_stubs = {}
        self._stub_targets = {
            "nodes": {},
            "edges": {}
        }

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
        :param kwargs: other valid args: `target`, the target of the rpc server to fetch node
                        attributes, see :class:`graph2tensor.client.distributed.DistNumpyAttributeStoreServicer`
                        for details
        """
        super(DistNumpyGraph, self).add_node(node_type, attrs_info, labeled)
        target = kwargs.get("target")
        if target:
            channel = grpc.insecure_channel(target)
            self._node_stubs[node_type] = DistNumpyAttributeStoreStub(channel)
        self._stub_targets["nodes"][node_type] = target

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
        :param kwargs: other valid args: `relation_target`, required, the target of rpc server
                        to fetch the relation data of edge, see
                        :class:`graph2tensor.client.distributed.DistNumpyRelationStoreServicer`
                        for details; `attribute_target`, the target of rpc server to fetch the
                        attributes data of edge, see
                        :class:`graph2tensor.client.distributed.DistNumpyAttributeStoreServicer`
                        for details
        """
        super(DistNumpyGraph, self).add_edge(edge_type, src_type, dst_type, attrs_info, directed)
        relation_target = kwargs["relation_target"]
        channel = grpc.insecure_channel(relation_target)
        self._edge_stubs[edge_type] = {}
        self._edge_stubs[edge_type]["relation"] = DistNumpyRelationStoreStub(channel)
        attribute_target = kwargs.get("attribute_target")
        if attribute_target:
            channel = grpc.insecure_channel(attribute_target)
            self._edge_stubs[edge_type]["attribute"] = DistNumpyAttributeStoreStub(channel)
        self._stub_targets["edges"][edge_type] = {
            "relation": relation_target,
            "attribute": attribute_target
        }

    def lookup_nodes(self, nodes, **kwargs):
        if nodes.node_type not in self._schema["nodes"]:
            raise ValueError("graph has no node type: %s" % nodes.node_type)
        if not self._schema["nodes"][nodes.node_type]["attrs"]:
            raise ValueError("node %s has no attribute to fetch" % nodes.node_type)
        request = NdarrayProto(
            dtype=nodes.ids.dtype.__str__(),
            array_content=nodes.ids.tobytes()
        )
        request.dims.extend(nodes.ids.shape)
        attrs = self._node_stubs[nodes.node_type].lookup(request).attrs
        attrs = {
            k: np.frombuffer(attrs[k].array_content, dtype=attrs[k].dtype).reshape(*attrs[k].dims)
            for k in attrs
        }
        return attrs

    def lookup_edges(self, edges, **kwargs):
        if edges.edge_type not in self._schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edges.edge_type)
        if not self._schema["edges"][edges.edge_type]["attrs"]:
            raise ValueError("edge %s has no attribute to fetch" % edges.edge_type)
        request = NdarrayProto(
            dtype=edges.edge_ids.dtype.__str__(),
            array_content=edges.edge_ids.tobytes()
        )
        request.dims.extend(edges.edge_ids.shape)
        attrs = self._edge_stubs[edges.edge_type]["attribute"].lookup(request).attrs
        attrs = {
            k: np.frombuffer(attrs[k].array_content, dtype=attrs[k].dtype).reshape(*attrs[k].dims)
            for k in attrs
        }
        return attrs

    def get_node_attr_data(self, node_type, attr_name, **kwargs):
        """
        Get the data for attribute `attr_name` of the node `node_type`.

        :param node_type: node type
        :param attr_name: attribute name
        :param kwargs: other valid args: `batch_size`, the size of mini-batch, default to 4096
        :return: a generator which yield the attribute data in np.ndarray format
        """
        if node_type not in self.schema["nodes"]:
            raise ValueError("graph has no node type: %s" % node_type)
        if attr_name != "label" and attr_name not in self.get_node_attr_info(node_type):
            raise ValueError("node %s has no attribute %s" % (node_type, attr_name))
        request = AttrRequest(name=attr_name, batch_size=kwargs.get("batch_size", 4096))
        stream = self._node_stubs[node_type].get_attr_data(request)
        for x in stream:
            yield np.frombuffer(x.array_content, dtype=x.dtype).reshape(*x.dims)

    def get_node_label(self, node_type, **kwargs):
        """
        Get the label of node `node_type`

        :param node_type: node type
        :param kwargs: other valid args: `batch_size`, the size of mini-batch, default to 4096
        :return: a generator which yield node label in np.ndarray format
        """
        if node_type not in self.schema["nodes"]:
            raise ValueError("graph has no node type: %s" % node_type)
        if not self.is_node_labeled(node_type):
            raise ValueError("node %s was not labeled" % (node_type,))
        return self.get_node_attr_data(node_type, "label", **kwargs)

    def get_edge_attr_data(self, edge_type, attr_name, **kwargs):
        """
        Get the data for attribute `attr_name` of the edge `edge_type`

        :param edge_type: edge type
        :param attr_name: attribute name
        :param kwargs: other valid args: `batch_size`, the size of mini-batch, default to 4096
        :return: a generator which yield the attribute data in np.ndarray format
        """
        if edge_type not in self.schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edge_type)
        if attr_name not in self.get_edge_attr_info(edge_type):
            raise ValueError("edge %s has no attribute %s" % (edge_type, attr_name))
        request = AttrRequest(name=attr_name, batch_size=kwargs.get("batch_size", 4096))
        stream = self._edge_stubs[edge_type]["attribute"].get_attr_data(request)
        for x in stream:
            yield np.frombuffer(x.array_content, dtype=x.dtype).reshape(*x.dims)

    def get_edge_ids(self, edge_type, **kwargs):
        """
        Get src_ids, dst_ids and edge_ids of all the edges with specified type.

        :param edge_type: edge type
        :param kwargs: other valid args: `batch_size`, the size of mini-batch, default to 4096
        :return: a generator which yield src_ids, dst_ids and edge_ids,
                all in `np.ndarray` format
        """
        if edge_type not in self.schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edge_type)
        request = BatchingProto(size=kwargs.get("batch_size", 4096))
        stream = self._edge_stubs[edge_type]["relation"].get_edge_ids(request)
        for x in stream:
            yield np.frombuffer(x.src_ids.array_content, dtype=x.src_ids.dtype).reshape(*x.src_ids.dims), \
                np.frombuffer(x.dst_ids.array_content, dtype=x.dst_ids.dtype).reshape(*x.dst_ids.dims), \
                np.frombuffer(x.edge_ids.array_content, dtype=x.edge_ids.dtype).reshape(*x.edge_ids.dims)

    def get_edge_probs(self, edge_type, **kwargs):
        """
        Get the edge probabilities of all the edges with specified type.

        :param edge_type: edge type
        :param kwargs: other valid args: `batch_size`, the size of mini-batch, default to 4096
        :return: a generator which yield the edge probabilities in `np.ndarray` format
        """
        if edge_type not in self.schema["edges"]:
            raise ValueError("graph has no edge type: %s" % edge_type)
        request = BatchingProto(size=kwargs.get("batch_size", 4096))
        stream = self._edge_stubs[edge_type]["relation"].get_edge_probs(request)
        for x in stream:
            yield np.frombuffer(x.array_content, dtype=x.dtype).reshape(*x.dims)

    def sample_neighbors(self, ids, edge_type, num=None, strategy="random", **kwargs):
        """
        Sample neighbors along with the specified edge according to the specified strategy.

        :param ids: the id of src or dst(when `reverse` is `True`) nodes
        :param edge_type: the type of the edge to sample along with
        :param num: the number of neighbors to sample for each node
        :param strategy: sampling strategy, several strategies are supported and default as "random".
                        "random": sample `num` neighbors randomly with/without replacement,
                        "all": sample all neighbors and ignore the `num` param,
                        "topk": sample `num` neighbors with highest edge weights,
                        return all neighbors if not enough.
        :param kwargs: other valid args: `num_threads`, specify the number of working threads;
                       `replace`, whether the sample is with or without replacement, only valid
                       for "random" sampling strategy;`use_edge_probs`, whether use edge probabilities
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

        request = SamplingRequest(
            ids=NdarrayProto(dtype=ids.dtype.__str__(), array_content=ids.tobytes()),
            n=num if num is not None else 0,
            num_threads=kwargs.get("num_threads", -1),
            replace=kwargs.get("replace", True),
            use_edge_probs=kwargs.get("use_edge_probs", False)
        )
        request.ids.dims.extend(ids.shape)
        if strategy.upper() == "ALL":
            hop = self._edge_stubs[edge_type]["relation"].sample_all_neighbors(request)
        elif strategy.upper() == "TOPK":
            hop = self._edge_stubs[edge_type]["relation"].sample_topk_neighbors(request)
        else:
            hop = self._edge_stubs[edge_type]["relation"].sample_neighbors_randomly(request)

        nbr_ids = np.frombuffer(
            hop.nbr_ids.array_content,
            dtype=hop.nbr_ids.dtype).reshape(*hop.nbr_ids.dims)
        offset = np.frombuffer(
            hop.offset.array_content,
            dtype=hop.offset.dtype).reshape(*hop.offset.dims)
        edge_ids = np.frombuffer(
            hop.edge_ids.array_content,
            dtype=hop.edge_ids.dtype).reshape(*hop.edge_ids.dims)
        edges = Edges(edge_type, edge_ids=edge_ids, src_ids=np.repeat(ids.ravel(), offset), dst_ids=nbr_ids)
        return Nodes(dst_type, nbr_ids, offset), edges

    def to_config(self):
        return {
            "stub_targets": self._stub_targets,
            "schema": self.schema
        }

    @classmethod
    def from_config(cls, config):
        obj = cls()
        obj._schema = config["schema"]
        obj._stub_targets = config["stub_targets"]
        for node_type, target in obj._stub_targets["nodes"].items():
            if target:
                channel = grpc.insecure_channel(target)
                obj._node_stubs[node_type] = DistNumpyAttributeStoreStub(channel)
        for edge_type, targets in obj._stub_targets["edges"].items():
            channel = grpc.insecure_channel(targets["relation"])
            obj._edge_stubs[edge_type] = {}
            obj._edge_stubs[edge_type]["relation"] = DistNumpyRelationStoreStub(channel)
            if targets["attribute"]:
                channel = grpc.insecure_channel(targets["attribute"])
                obj._edge_stubs[edge_type]["attribute"] = DistNumpyAttributeStoreStub(channel)
        return obj

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
        request = WalkRequest(
            ids=NdarrayProto(dtype=ids.dtype.__str__(), array_content=ids.tobytes()),
            walk_length=walk_length,
            use_edge_probs=kwargs.get("use_edge_probs", False),
            discard_frequent_nodes=kwargs.get("discard_frequent_nodes", True),
            freq_th=kwargs.get("freq_th", 1e-5)
        )
        request.ids.dims.extend(ids.shape)
        paths = self._edge_stubs[edge_type]["relation"].random_walk(request)
        return tuple(json.loads(paths.paths))

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
        request = WalkRequest(
            ids=NdarrayProto(dtype=ids.dtype.__str__(), array_content=ids.tobytes()),
            walk_length=walk_length,
            p=p,
            q=q,
            use_edge_probs=kwargs.get("use_edge_probs", False),
            discard_frequent_nodes=kwargs.get("discard_frequent_nodes", True),
            freq_th=kwargs.get("freq_th", 1e-5)
        )
        request.ids.dims.extend(ids.shape)
        paths = self._edge_stubs[edge_type]["relation"].node2vec_walk(request)
        return tuple(json.loads(paths.paths))

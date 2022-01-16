#!/usr/bin/env python3
import dist_numpy_relation_store_pb2_grpc
from dist_numpy_relation_store_pb2 import EdgeIdsProto, HopProto, PathsProto
from ndarray_pb2 import NdarrayProto
import numpy as np
import math
import grpc
import json
from .._numpy_store import _NumpyRelationStore


class DistNumpyRelationStoreServicer(dist_numpy_relation_store_pb2_grpc.DistNumpyRelationStoreServicer):
    """
    The server for edge relation data fetching.

    :param src_ids: the ids of source nodes in `np.ndarray` format
    :param dst_ids: the ids of destination nodes in `np.ndarray` format
    :param edge_ids: the ids of edges, if not specified, a sequence from
                    0 to number of edges will be generated as edges ids
    :param edge_probs: the probabilities of edges, when sample neighbours
                       along the edges using "topk" strategy, `edge_probs`
                       is required and will be used to computed the topk neighbours,
                       for "random" sampling strategy, `edge_probs` is optional,
                       if specified, neighbours will be sampled based on edge
                       probabilities, otherwise sampled uniformly
    :param directed: whether edge is directed, if not, an reversed edge will also be added.
    """

    def __init__(self, src_ids, dst_ids, edge_ids=None, edge_probs=None, directed=True):
        super(DistNumpyRelationStoreServicer, self).__init__()
        self._relation_store = _NumpyRelationStore()
        self._relation_store.add_edge(
            "", directed, src_ids, dst_ids, edge_ids=edge_ids, edge_probs=edge_probs
        )

    def get_edge_ids(self, request, context):
        src_ids, dst_ids, edge_ids = self._relation_store.get_edge_ids("")
        total_n = src_ids.shape[0]
        for i in range(math.ceil(total_n/request.size)):
            x = src_ids[i*request.size:(i+1)*request.size]
            y = dst_ids[i*request.size:(i+1)*request.size]
            z = edge_ids[i*request.size:(i+1)*request.size]
            eids_proto = EdgeIdsProto(
                src_ids=NdarrayProto(dtype=x.dtype.__str__(), array_content=x.tobytes()),
                dst_ids=NdarrayProto(dtype=y.dtype.__str__(), array_content=y.tobytes()),
                edge_ids=NdarrayProto(dtype=z.dtype.__str__(), array_content=z.tobytes()),
            )
            eids_proto.src_ids.dims.extend(x.shape)
            eids_proto.dst_ids.dims.extend(y.shape)
            eids_proto.edge_ids.dims.extend(z.shape)
            yield eids_proto

    def get_edge_probs(self, request, context):
        edge_probs = self._relation_store.get_edge_probs("")
        if edge_probs is None:
            raise grpc.RpcError("edge has no probability")
        total_n = edge_probs.shape[0]
        for i in range(math.ceil(total_n/request.size)):
            array_content = edge_probs[i*request.size:(i+1)*request.size]
            x = NdarrayProto(
                dtype=array_content.dtype.__str__(),
                array_content=array_content.tobytes())
            x.dims.extend(array_content.shape)
            yield x

    @staticmethod
    def _build_hop_proto(nbr_ids, offset, edge_ids=None):
        if edge_ids is not None:
            hop = HopProto(
                nbr_ids=NdarrayProto(dtype=nbr_ids.dtype.__str__(), array_content=nbr_ids.tobytes()),
                offset=NdarrayProto(dtype=offset.dtype.__str__(), array_content=offset.tobytes()),
                edge_ids=NdarrayProto(dtype=edge_ids.dtype.__str__(), array_content=edge_ids.tobytes())
            )
            hop.edge_ids.dims.extend(edge_ids.shape)
        else:
            hop = HopProto(
                nbr_ids=NdarrayProto(dtype=nbr_ids.dtype.__str__(), array_content=nbr_ids.tobytes()),
                offset=NdarrayProto(dtype=offset.dtype.__str__(), array_content=offset.tobytes()),
            )
        hop.nbr_ids.dims.extend(nbr_ids.shape)
        hop.offset.dims.extend(offset.shape)
        return hop

    def sample_all_neighbors(self, request, context):
        ids = np.frombuffer(
            request.ids.array_content,
            dtype=request.ids.dtype
        ).reshape(*request.ids.dims)
        nbr_ids, offset, edge_ids = self._relation_store.sample_all_neighbors(ids, "")
        hop = self._build_hop_proto(nbr_ids, offset, edge_ids)
        return hop

    def sample_topk_neighbors(self, request, context):
        ids = np.frombuffer(request.ids.array_content, dtype=request.ids.dtype).reshape(*request.ids.dims)
        nbr_ids, offset, edge_ids = self._relation_store.sample_topk_neighbors(
            ids, "", request.n, num_threads=request.num_threads
        )
        hop = self._build_hop_proto(nbr_ids, offset, edge_ids)
        return hop

    def sample_neighbors_randomly(self, request, context):
        ids = np.frombuffer(request.ids.array_content, dtype=request.ids.dtype).reshape(*request.ids.dims)
        nbr_ids, offset, edge_ids = self._relation_store.sample_neighbors_randomly(
            ids, "", request.n,
            replace=request.replace,
            num_threads=request.num_threads,
            use_edge_probs=request.use_edge_probs
        )
        hop = self._build_hop_proto(nbr_ids, offset, edge_ids)
        return hop

    def random_walk(self, request, context):
        ids = np.frombuffer(request.ids.array_content, dtype=request.ids.dtype).reshape(*request.ids.dims)
        paths = self._relation_store.random_walk(
            ids, "", request.walk_length,
            use_edge_probs=request.use_edge_probs,
            discard_frequent_nodes=request.discard_frequent_nodes,
            freq_th=request.freq_th
        )
        return PathsProto(paths=json.dumps(paths))

    def node2vec_walk(self, request, context):
        ids = np.frombuffer(request.ids.array_content, dtype=request.ids.dtype).reshape(*request.ids.dims)
        paths = self._relation_store.node2vec_walk(
            ids, "", request.walk_length, request.p, request.q,
            use_edge_probs=request.use_edge_probs,
            discard_frequent_nodes=request.discard_frequent_nodes,
            freq_th=request.freq_th
        )
        return PathsProto(paths=json.dumps(paths))

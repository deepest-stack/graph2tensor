#!/usr/bin/env python3
import numpy as np
from .base_converter import BaseConverter
from ..egograph import EgoGraph
from typing import Tuple, Dict


NID = '_NID'
EID = '_EID'
UID = '_UID'
VID = '_VID'


class Ego2Tensor(BaseConverter):
    """
    An object that convert ego-graph into tensor.

    :param graph: the graph to lookup nodes & edges attributes
    :param include_edge: whether or not lookup edges attributes
    """

    def __init__(self, graph, include_edge=False):
        self._graph = graph
        self._include_edge = include_edge

    def convert(self, egograph: EgoGraph, **kwargs) \
            -> Tuple[Tuple[Tuple[Dict[str, np.ndarray],
                                 Dict[str, np.ndarray],
                                 Dict[str, np.ndarray], np.ndarray], ...], ...]:
        """
        Fetch weight & attributes of nodes and edges in ego-graph and
        convert the ego-graph into format that could be feed into tensorflow

        :param egograph: :class:`graph2tensor.egograph.EgoGraph` object
        :return: A tuple representation of sub-graph as (path#1, path#2, ..., path#N).
                `path#N` is a tuple, representing a path in ego-graph, whose element -- `hop`
                is triplet as (src, edge, dst, offset) and `src`/`edge`/`dst` is a dict
                with attribute name as key and attribute in `numpy.ndarray` format as value.
                Besides normal attributes, there are several reserved keys in node & edge attributes dict, for node
                attributes there will always be a reserved key named `graph2tensor.NID` which store the
                ids of nodes, for edge attributes, there will be 3 reserved keys, `graph2tensor.EID` - ids of edges,
                `graph2tensor.UID` - source ids of edges and `graph2tensor.VID` - destination ids of edges.
                `segment_ids` is a `np.1darray`, it record the match-up from `dst` & `edge` to `src`,
                `[0, 0, 1, 2, 2, 3, 3, 3]` means the 1st & 2nd `dst` node & `edge` belong to 1st `src`
                node, 3rd `dst` node & `edge` to 2nd `src`, 4th & 5th `dst` node & `edge` to 3rd `src`,
                6th, 7th and 8th `dst` node & `edge` to 4th `src`. see
                `Segmentation <https://www.tensorflow.org/api_docs/python/tf/math#Segmentation>`__ for
                more details.

                e.g.

                .. code-block::

                    (
                        # path#1
                        ((src, edge, dst, segment_ids), #hop#1
                        (src, edge, dst, segment_ids),  #hop#2
                        (src, edge, dst, segment_ids)), #hop#3
                        # path#2
                        ((src, edge, dst, segment_ids), #hop#1
                        (src, edge, dst, segment_ids),  #hop#2
                        (src, edge, dst, segment_ids)), #hop#3
                        # path#N
                        ...
                    )
        """
        rst = []
        centre_attrs = self._graph.lookup_nodes(egograph.centre_nodes)
        centre_attrs[NID] = egograph.centre_nodes.ids
        for path in egograph.paths:
            hops = []
            dst_attrs = centre_attrs
            for edge, dst_node in path:
                edge_attrs = {
                    EID: edge.edge_ids,
                    UID: edge.src_ids,
                    VID: edge.dst_ids
                }
                if self._include_edge:
                    edge_attrs.update(self._graph.lookup_edges(edge))
                src_attrs = dst_attrs
                dst_attrs = self._graph.lookup_nodes(dst_node)
                dst_attrs[NID] = dst_node.ids
                segment_ids = np.repeat(np.arange(dst_node.offset.shape[0]), dst_node.offset)
                hops.append((src_attrs, edge_attrs, dst_attrs, segment_ids))
            rst.append(tuple(hops))
        return tuple(rst)

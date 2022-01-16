#!/usr/bin/env python3
from typing import Sequence, Tuple, Dict, Union, Optional, Any, List
from ..common import Nodes, Edges
import numpy as np


class BaseGraph(object):
    """
    The base class for graph data manipulating.
    """

    def __init__(self):
        self._schema = {"nodes": {}, "edges": {}}

    def add_node(self,
                 node_type: str,
                 attrs_info: Union[Sequence[Tuple[str, str]], Dict[str, str]],
                 labeled: bool,
                 **kwargs):
        """
        Add a new node into graph

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
        """
        self._schema["nodes"][node_type] = {"labeled": labeled}
        if isinstance(attrs_info, dict):
            self._schema["nodes"][node_type]["attrs"] = attrs_info
        elif isinstance(attrs_info, list) or isinstance(attrs_info, tuple):
            self._schema["nodes"][node_type]["attrs"] = dict(attrs_info)
        else:
            raise ValueError("`node_attrs` expected `dict` or `list` or `tuple`, but received %s"
                             % attrs_info.__class__.__name__)

    def add_edge(self,
                 edge_type: str,
                 src_type: str,
                 dst_type: str,
                 attrs_info: Union[Sequence[Tuple[str, str]], Dict[str, str]],
                 directed: bool = True,
                 **kwargs):
        """
        Add a new edge into graph

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
                            default to `True`
        """
        if not directed and src_type != dst_type:
            raise ValueError("undirected edge should have same src & dst node type")
        self._schema["edges"][edge_type] = {}
        self._schema["edges"][edge_type]["src_type"] = src_type
        self._schema["edges"][edge_type]["dst_type"] = dst_type
        self._schema["edges"][edge_type]["directed"] = directed
        if isinstance(attrs_info, dict):
            self._schema["edges"][edge_type]["attrs"] = attrs_info
        elif isinstance(attrs_info, list) or isinstance(attrs_info, tuple):
            self._schema["edges"][edge_type]["attrs"] = dict(attrs_info)
        else:
            _ = self._schema["edges"].pop(edge_type)
            raise ValueError("`edge_attrs` expected `dict` or `list` or `tuple`, but received %s"
                             % attrs_info.__class__.__name__)

    def to_config(self) -> Dict[str, Any]:
        """
        Serialize object into dict.

        :return: a dict which could be used to deserialize
        """
        raise NotImplemented

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Deserialize object from dict.

        :param config: config in `dict` format to deserialized from
        :return: an instance of `cls` class
        """
        raise NotImplemented

    @property
    def schema(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Return the schema of graph.

        :return: schema of graph

        :Example:

        An example of graph schema

        >>> from graph2tensor.client import BaseGraph
        >>> import json
        >>> graph = BaseGraph()
        >>> graph.add_node(
        ...     "employee",
        ...     [("name", "str"), ("age", "int"), ("dept", "str"), ("salary", "float")],
        ...     True)
        >>> graph.add_node(
        ...     "dept",
        ...     [("num_employee", "int"), ("name", "str")],
        ...     False)
        >>> graph.add_edge(
        ...     "workmate",
        ...     "employee",
        ...     "employee",
        ...     [],
        ...     False)
        >>> graph.add_edge(
        ...     "is_header",
        ...     "employee",
        ...     "dept",
        ...     [("since", "str"),],
        ...     True)
        >>> print(json.dumps(graph.schema, indent=4))
        {
            "nodes": {
                "employee": {
                    "labeled": true,
                    "attrs": {
                        "name": "str",
                        "age": "int",
                        "dept": "str",
                        "salary": "float"
                    }
                },
                "dept": {
                    "labeled": false,
                    "attrs": {
                        "num_employee": "int",
                        "name": "str"
                    }
                }
            },
            "edges": {
                "workmate": {
                    "src_type": "employee",
                    "dst_type": "employee",
                    "directed": false,
                    "attrs": {}
                },
                "is_header": {
                    "src_type": "employee",
                    "dst_type": "dept",
                    "directed": true,
                    "attrs": {
                        "since": "str"
                    }
                }
            }
        }

        """
        return self._schema

    def lookup_nodes(self, nodes: Nodes, **kwargs) -> Dict[str, np.ndarray]:
        """
        Lookup all the attributes of the given nodes.

        :param nodes: a :class:`graph2tensor.common.Nodes` object, nodes to lookup
        :return: a dict whose keys are attributes name and values are
                 attributes value in numpy.ndarray format
        """
        raise NotImplemented

    def lookup_edges(self, edges: Edges, **kwargs) -> Dict[str, np.ndarray]:
        """
        Lookup all the attributes of the given edges.

        :param edges: a :class:`graph2tensor.common.Edges` object, edges to lookup
        :return: a dict whose keys are attributes name and values are
                 attributes value in numpy.ndarray format
        """
        raise NotImplemented

    def _get_edge_info(self, edge_type: str, info_key: str) -> Any:
        if edge_type not in self._schema["edges"]:
            raise ValueError("Edge type %s not exists" % (edge_type, ))
        return self._schema["edges"][edge_type][info_key]

    def get_src_type(self, edge_type: str) -> str:
        """
        Get the type of source node of given edge type.

        :param edge_type: edge type
        :return: the type of source node of given edge type
        """
        return self._get_edge_info(edge_type, "src_type")

    def get_dst_type(self, edge_type: str) -> str:
        """
        Get the type of destination node of given edge type.

        :param edge_type: edge type
        :return: the type of destination node of given edge type
        """
        return self._get_edge_info(edge_type, "dst_type")

    def get_edge_attr_info(self, edge_type: str) -> Dict[str, str]:
        """
        Get the attribute name & type of given edge type.

        :param edge_type: edge type
        :return: a dict whose keys are attributes name and values
                 are attributes type
        """
        return self._get_edge_info(edge_type, "attrs")

    def get_edge_attr_data(self, edge_type: str, attr_name: str, **kwargs) -> np.ndarray:
        """
        Get the data for attribute `attr_name` of the edge `edge_type`

        :param edge_type: edge type
        :param attr_name: attribute name
        :return: the attribute data in np.ndarray format
        """
        raise NotImplemented

    def is_edge_directed(self, edge_type: str) -> bool:
        """
        Check whether the edge is directed.

        :param edge_type: edge type
        :return: 'True` if edge is directed else `False`
        """
        return self._schema["edges"][edge_type]["directed"]

    def _get_node_info(self, node_type: str, info_key: str) -> Any:
        if node_type not in self._schema["nodes"]:
            raise ValueError("Node type %s not exists" % (node_type,))
        return self._schema["nodes"][node_type][info_key]

    def get_node_attr_info(self, node_type: str) -> Dict[str, str]:
        """
        Get the attribute name & type of given node type.

        :param node_type: node type
        :return: a dict whose keys are attributes name and values
                 are attributes type
        """
        return self._get_node_info(node_type, "attrs")

    def is_node_labeled(self, node_type: str) -> bool:
        """
        Check whether the node is labeled.

        :param node_type: node type
        :return: `True` if node is labeled else `False`
        """
        return self._get_node_info(node_type, "labeled")

    def get_node_attr_data(self, node_type: str, attr_name: str, **kwargs) -> np.ndarray:
        """
        Get the data for attribute `attr_name` of the node `node_type`.

        :param node_type: node type
        :param attr_name: attribute name
        :return: the attribute data in np.ndarray format
        """
        raise NotImplemented

    def get_node_label(self, node_type: str, **kwargs) -> np.ndarray:
        """
        Get the label of node `node_type`

        :param node_type: node type
        :return: node label in np.ndarray format
        """
        raise NotImplemented

    def sample_neighbors(self, ids: np.ndarray, edge_type: str,
                         num: Optional[int] = None, strategy: str = "random", **kwargs) -> Tuple[Nodes, Optional[Edges]]:
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
        :return: A `Nodes` object for neighbors, an `Edges` object or `None` for edges.
        """
        raise NotImplemented

    def random_walk(self, ids: np.ndarray, edge_type: str,
                    walk_length: int, **kwargs) -> Tuple[List[int], ...]:
        """
        Walk `walk_length` steps from `ids` along with edge `edge_type` randomly.

        :param ids: the id of nodes to start walk from
        :param edge_type: the type of edge to walk along
        :param walk_length: the length to walk
        :return: tuple whose element is list which representing a walk path
        """
        raise NotImplemented

    def node2vec_walk(self, ids: np.ndarray, edge_type: str,
                      walk_length: int, p: float, q: float, **kwargs) -> Tuple[List[int], ...]:
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
        :return: tuple whose element is list which representing a walk path
        """
        raise NotImplemented

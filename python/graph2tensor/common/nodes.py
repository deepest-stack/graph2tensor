#!/usr/bin/env python3
import numpy as np
import json
from typing import Sequence, Union


class Nodes(object):
    """
    The abstraction of a mini-batch of nodes.

    :param node_type: the type of the nodes
    :param ids: the ids of the nodes
    :param offset: the offsets of this nodes, indicating which nodes are
                   the neighbours of same node, default to None which will
                   lead the offset being initialized as a numpy.array with
                   same size as ids whose entries are all 1.

    :Example:

    In this example, nodes (1,), (2, 3), (4,5,6), (7, 8, 9, 10) belong to
    the neighbours of same node respectively.

    >>> from graph2tensor.common import Nodes
    >>> ids = [1,2,3,4,5,6,7,8,9,10]
    >>> offset = [1,2,3,4]
    >>> nodes = Nodes("nodeA", ids, offset)

    """

    def __init__(self,
                 node_type: str,
                 ids: Union[np.ndarray, Sequence[int]],
                 offset: Union[np.ndarray, Sequence[int], None] = None):
        self._node_type = node_type
        self._ids = np.ravel(ids)
        self._offset = np.ones_like(self._ids, dtype=np.int32) if offset is None \
            else np.array(offset, dtype=np.int32).ravel()

    @property
    def node_type(self) -> str:
        """
        Return the type of the nodes.

        :return: the type of the nodes
        """
        return self._node_type

    @property
    def ids(self) -> np.ndarray:
        """
        Return the ids of the nodes.

        :return: the ids of the nodes in numpy.ndarray format
        """
        return self._ids

    @property
    def offset(self) -> np.ndarray:
        """
        Return the offset of the nodes.

        :return: the offset of the nodes in numpy.ndarray format
        """
        return self._offset

    def __str__(self) -> str:
        return json.dumps(
            {
                "node_type": str(self._node_type),
                "ids": str(self._ids),
                "offset": str(self._offset)
            },
            indent=4
        )

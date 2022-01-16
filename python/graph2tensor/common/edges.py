#!/usr/bin/env python3
import numpy as np
import json
from typing import Union, Sequence, Optional


class Edges(object):
    """
    The abstraction of a mini-batch of edges.

    :param edge_type: the type of edges
    :param edge_ids: the ids of edges
    :param src_ids: the ids of source nodes
    :param dst_ids: the ids of destination nodes

    .. note::
       Edges could be characterized by src & dst node ids, or edge ids,
       or both. Thus, at least one of `src_ids` - `dst_ids` pair and
       `edge_ids` should be specified.

    :Example:
    >>> from graph2tensor.common import Edges
    >>> import numpy as np
    >>> src_ids = np.random.randint(10, size=20)
    >>> dst_ids = np.random.randint(10, size=20)
    >>> edge_ids = np.arange(20)
    >>> edges1 = Edges("edges1", edge_ids)
    >>> edges2 = Edges("edges2", src_ids=src_ids, dst_ids=dst_ids)
    >>> edges3 = Edges("edges3", edge_ids, src_ids, dst_ids)
    >>> print(edges1)
    {
        "edge_type": "edges1",
        "edge_ids": "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]",
        "src_ids": "None",
        "dst_ids": "None"
    }
    >>> print(edges2)
    {
        "edge_type": "edges2",
        "edge_ids": "None",
        "src_ids": "[0 3 6 5 4 7 4 9 8 4 6 4 2 6 4 6 6 8 6 2]",
        "dst_ids": "[2 0 7 6 1 2 8 0 5 9 6 2 9 6 8 2 0 9 6 9]"
    }
    >>> print(edges3)
    {
        "edge_type": "edges3",
        "edge_ids": "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]",
        "src_ids": "[0 3 6 5 4 7 4 9 8 4 6 4 2 6 4 6 6 8 6 2]",
        "dst_ids": "[2 0 7 6 1 2 8 0 5 9 6 2 9 6 8 2 0 9 6 9]"
    }

    """

    def __init__(self,
                 edge_type: str,
                 edge_ids: Union[np.ndarray, Sequence[int], None] = None,
                 src_ids: Union[np.ndarray, Sequence[int], None] = None,
                 dst_ids: Union[np.ndarray, Sequence[int], None] = None):
        self._edge_type = edge_type
        self._edge_ids = edge_ids
        if self._edge_ids is not None:
            self._edge_ids = np.ravel(self._edge_ids)
        self._src_ids = src_ids
        if self._src_ids is not None:
            self._src_ids = np.ravel(self._src_ids)
        self._dst_ids = dst_ids
        if self._dst_ids is not None:
            self._dst_ids = np.ravel(self._dst_ids)
        if self._src_ids is not None and self._dst_ids is None:
            raise ValueError("dst_ids missed")
        if self._dst_ids is not None and self._src_ids is None:
            raise ValueError("src_ids missed")
        if self._edge_ids is None and self._src_ids is None and self._dst_ids is None:
            raise ValueError("At least one of `edge_ids` and `src_ids`-`dst_ids` pair should be specified")

    @property
    def edge_type(self) -> str:
        """
        Get the type of edges.

        :return: the type of edges
        """
        return self._edge_type

    @property
    def edge_ids(self) -> Optional[np.ndarray]:
        """
        Get the ids of edges.

        :return: the ids of edges in numpy.ndarray format
        """
        return self._edge_ids

    @property
    def src_ids(self) -> Optional[np.ndarray]:
        """
        Get the ids of source nodes.

        :return: the ids of source nodes in numpy.ndarray format
        """
        return self._src_ids

    @property
    def dst_ids(self) -> Optional[np.ndarray]:
        """
        Get the ids of destination nodes.

        :return: the ids of destination nodes in numpy.ndarray format
        """
        return self._dst_ids

    def __str__(self) -> str:
        return json.dumps(
            {
                "edge_type": str(self._edge_type),
                "edge_ids": str(self._edge_ids),
                "src_ids": str(self._src_ids),
                "dst_ids": str(self._dst_ids)
            },
            indent=4
        )

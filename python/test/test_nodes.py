#!/usr/bin/env python3

from graph2tensor.common import Nodes
from unittest import TestCase, main
import numpy as np


class TestNodes(TestCase):

    def test_ids(self):
        nodes_a = Nodes('NodeA', np.arange(1, 101))
        assert nodes_a.ids.shape == (100, )
        nodes_b = Nodes('NodeB', np.reshape(np.arange(1, 101), newshape=(-1, 10)))
        assert nodes_b.ids.shape == (100,)

    def test_node_type(self):
        nodes = Nodes('NodeA', np.arange(1, 101))
        assert nodes.node_type == "NodeA"

    def test_offset(self):
        nodes = Nodes('NodeA', np.arange(1, 101))
        np.testing.assert_array_equal(nodes.offset, np.ones(shape=(100, ), dtype=np.int32))

    def test_str(self):
        nodes = Nodes('NodeA', np.arange(1, 101))
        print(nodes)


if __name__ == "__main__":
    main()

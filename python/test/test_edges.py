#!/usr/bin/env python3

from graph2tensor.common import Edges
from unittest import TestCase, main
import numpy as np


class TestEdges(TestCase):

    def test_edge(self):
        edge_a = Edges("edgeA", src_ids=np.arange(1, 101), dst_ids=np.arange(101, 201))
        with self.assertRaises(ValueError):
            edge_d = Edges("edgeD")
        with self.assertRaises(ValueError):
            edge_e = Edges("edgeE", np.arange(100), src_ids=np.arange(1, 101))
        with self.assertRaises(ValueError):
            edge_f = Edges("edgeF", np.arange(100), dst_ids=np.arange(1, 101))

    def test_edge_type(self):
        edge = Edges("edgeA", src_ids=np.arange(1, 101), dst_ids=np.arange(101, 201))
        assert edge.edge_type == "edgeA"

    def test_ids(self):
        edge_a = Edges("edgeA", src_ids=np.arange(1, 101), dst_ids=np.arange(101, 201))
        assert edge_a.src_ids.shape == (100, )
        assert edge_a.dst_ids.shape == (100, )
        edge_b = Edges("edgeB",
                       src_ids=np.arange(1, 101).reshape((-1, 10)),
                       dst_ids=np.arange(101, 201).reshape((-1, 10)))
        assert edge_b.src_ids.shape == (100, )
        assert edge_b.dst_ids.shape == (100, )
        edge_c = Edges("edgeC", edge_ids=np.arange(100))
        assert edge_c.edge_ids.shape == (100, )

    def test_str(self):
        edge = Edges("edgeA", src_ids=np.arange(1, 101), dst_ids=np.arange(101, 201))
        print(edge)


if __name__ == "__main__":
    main()

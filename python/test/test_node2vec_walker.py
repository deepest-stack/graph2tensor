#!/usr/bin/env python3

from graph2tensor.sampler import Node2VecWalker
from graph2tensor.client import NumpyGraph
from unittest import TestCase, main
from math import ceil
from test_utils import graph_setup
g = NumpyGraph()


class TestNode2VecWalker(TestCase):

    def test_1_setup(self) -> None:
        graph_setup(g)

    def test_node2vec_walk(self):
        walker = Node2VecWalker(g, edge_type='cites', walk_length=4, p=2.0, q=0.5)
        ids, _, _ = g.get_edge_ids('cites')
        for i in range(ceil(ids.shape[0]//1024)):
            x = walker.sample(ids[i*1024:(i+1)*1024])
            # print(x)


if __name__ == "__main__":
    main()

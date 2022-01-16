#!/usr/bin/env python3

from graph2tensor.sampler import RandomWalker
from graph2tensor.client import NumpyGraph
from unittest import TestCase, main
from math import ceil
from test_utils import graph_setup
g = NumpyGraph()


class TestRandomWalker(TestCase):

    def test_1_setup(self) -> None:
        graph_setup(g)

    def test_random_walk(self):
        walker = RandomWalker(g, edge_type='cites', walk_length=4)
        ids, _, _ = g.get_edge_ids('cites')
        for i in range(ceil(ids.shape[0]//1024)):
            _ = walker.sample(ids[i*1024:(i+1)*1024])


if __name__ == "__main__":
    main()

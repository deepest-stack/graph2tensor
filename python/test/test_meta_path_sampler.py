#!/usr/bin/env python3

from graph2tensor.sampler import MetaPathSampler
from graph2tensor.client import NumpyGraph
from unittest import TestCase, main
import numpy as np
from test_utils import graph_setup
g = NumpyGraph()


class TestSampler(TestCase):

    def test_1_setup(self) -> None:
        graph_setup(g)

    def _test_sample(self, strategy, **kwargs):
        sampler = MetaPathSampler(g, ["(paper) -[cites]- (paper) -[cites]- (paper)"], 3, strategy)
        ids = np.random.randint(169343, size=1024)
        sub_g = sampler.sample(ids, **kwargs)
        assert sub_g.centre_nodes.ids.shape[0] == 1024
        assert sub_g.paths[0].__len__() == 2

    def test_random_sample(self):
        self._test_sample("random", replace=False)

    def test_topk_sample(self):
        self._test_sample("topk")

    def test_all_sample(self):
        self._test_sample("all")


if __name__ == "__main__":
    main()

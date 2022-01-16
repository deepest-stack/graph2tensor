#!/usr/bin/env python3

from graph2tensor.sampler import MetaPathSampler
from graph2tensor.client import NumpyGraph
from graph2tensor.converter import Ego2Tensor
from unittest import TestCase, main
import numpy as np
from test_utils import graph_setup
g = NumpyGraph()


class TestEgo2Tensor(TestCase):

    def test_1_setup(self):
        graph_setup(g)

    def test_ego2tensor(self):
        sampler = MetaPathSampler(g, ["(paper) -[cites]- (paper) -[cites]- (paper)"], 4, "random")
        converter = Ego2Tensor(g)
        for _ in range(10):
            ids = np.random.randint(169343, size=1024)
            sub_g = sampler.sample(ids, num_threads=8)
            x = converter.convert(sub_g)


if __name__ == "__main__":
    main()

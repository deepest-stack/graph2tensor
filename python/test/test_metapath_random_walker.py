#!/usr/bin/env python3
import numpy as np
from graph2tensor.sampler import MetaPathRandomWalker
from graph2tensor.client import NumpyGraph
from unittest import TestCase, main
from math import ceil
from test_utils import graph_setup
g = NumpyGraph()


class TestMetaPathRandomWalker(TestCase):

    def test_1_setup(self) -> None:
        graph_setup(g)

    def test_metapath_random_walk(self):
        walker = MetaPathRandomWalker(g, meta_path="(paper) -[cites]- (paper) -[cites]- (paper)", walk_length=6)
        ids = np.arange(169343)
        for i in range(ceil(ids.shape[0]//1024)):
            _ = walker.sample(ids[i*1024:(i+1)*1024])


if __name__ == "__main__":
    main()

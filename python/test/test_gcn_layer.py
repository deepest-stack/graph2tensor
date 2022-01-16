#!/usr/bin/env python3

from graph2tensor.model.layers import GCNConv
from unittest import TestCase, main
import numpy as np
import tensorflow as tf


class TestGCN(TestCase):

    def test_config(self):
        gcn = GCNConv(units=32, add_self_loop=True, name="gcn")
        config = gcn.get_config()
        assert config["units"] == 32
        assert config["add_self_loop"] is True
        assert config["name"] == "gcn"
        _ = GCNConv.from_config(config)

    def test_call(self):
        hop = (
            tf.constant(np.random.random(size=(10, 16)), dtype=tf.float64),
            tf.constant([]),
            tf.constant(np.random.random(size=(55, 16)), dtype=tf.float64),
            np.repeat(np.arange(10), np.arange(1, 11))
        )
        gcn = GCNConv(32, add_self_loop=False)
        assert gcn(hop).numpy().shape == (10, 32)


if __name__ == "__main__":
    main()

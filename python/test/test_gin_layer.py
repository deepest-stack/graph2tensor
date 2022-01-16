#!/usr/bin/env python3

from graph2tensor.model.layers import GINConv
from unittest import TestCase, main
import numpy as np
import tensorflow as tf


class TestGCN(TestCase):

    def test_config(self):
        gcn = GINConv(name="gin")
        config = gcn.get_config()
        assert config["name"] == "gin"
        _ = GINConv.from_config(config)

    def test_call(self):
        hop = (
            tf.constant(np.random.random(size=(10, 16)), dtype=tf.float64),
            tf.constant(np.random.random(size=(55, 1)), dtype=tf.float64),
            tf.constant(np.random.random(size=(55, 16)), dtype=tf.float64),
            np.repeat(np.arange(10), np.arange(1, 11))
        )
        gcn = GINConv(
            mlp_units=[32, 32],
            mlp_activations='tanh'
        )
        assert gcn(hop).numpy().shape == (10, 32)
        assert gcn(hop, edge_weighted=True).numpy().shape == (10, 32)


if __name__ == "__main__":
    main()

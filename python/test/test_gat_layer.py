#!/usr/bin/env python3

from graph2tensor.model.layers import GATConv
from unittest import TestCase, main
import numpy as np
import tensorflow as tf


class TestGAT(TestCase):

    def test_config(self):
        gat = GATConv(units=32, name="gat", multi_heads_reduction='concat')
        config = gat.get_config()
        assert config["units"] == 32
        assert config["num_heads"] == 1
        assert config["negative_slope"] == 0.2
        assert config["name"] == 'gat'
        _ = GATConv.from_config(config)

    def test_multi_heads_reduction(self):
        with self.assertRaises(Warning):
            _ = GATConv(units=32, name="gat", multi_heads_reduction='xxx')

    def test_call(self):
        hop = (
            tf.constant(np.random.random(size=(10, 16)), dtype=tf.float64),
            tf.constant([]),
            tf.constant(np.random.random(size=(55, 16)), dtype=tf.float64),
            np.repeat(np.arange(10), np.arange(1, 11))
        )
        gat = GATConv(32, num_heads=4, multi_heads_reduction='mean')
        assert gat(hop).numpy().shape == (10, 32)
        gat = GATConv(32, num_heads=4, multi_heads_reduction='concat')
        assert gat(hop).numpy().shape == (10, 32*4)
        y, attn = gat(hop, with_attention=True)
        attn_sum = tf.math.segment_sum(attn[:55], tf.repeat(np.arange(10), np.arange(1, 11))) + attn[55:]
        np.testing.assert_allclose(attn_sum, np.ones((10, 4)), rtol=1e-6)


if __name__ == "__main__":
    main()

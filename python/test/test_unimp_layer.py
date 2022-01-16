#!/usr/bin/env python3

from graph2tensor.model.layers import UniMP
from unittest import TestCase, main
import numpy as np
import tensorflow as tf


class TestUniMP(TestCase):

    def test_config(self):
        unimp = UniMP(units=32, num_heads=4, name="unimp", multi_heads_reduction='concat')
        config = unimp.get_config()
        assert config["units"] == 32
        assert config["num_heads"] == 4
        assert config["name"] == 'unimp'
        assert config["multi_heads_reduction"] == "concat"
        _ = UniMP.from_config(config)

    def test_multi_heads_reduction(self):
        with self.assertRaises(Warning):
            _ = UniMP(units=32, name="unimp", multi_heads_reduction='xxx')

    def test_call(self):
        hop = (
            tf.constant(np.random.random(size=(10, 16)), dtype=tf.float64),
            tf.constant(np.random.random(size=(55, 17)), dtype=tf.float64),
            tf.constant(np.random.random(size=(55, 18)), dtype=tf.float64),
            tf.repeat(np.arange(10), np.arange(1, 11))
        )
        unimp = UniMP(32, num_heads=4, multi_heads_reduction='mean')
        assert unimp(hop).numpy().shape == (10, 32)
        unimp = UniMP(32, num_heads=4, multi_heads_reduction='concat')
        assert unimp(hop).numpy().shape == (10, 32*4)
        y, attn = unimp(hop, with_attention=True)
        attn_sum = tf.math.segment_sum(attn, tf.repeat(np.arange(10), np.arange(1, 11), axis=0))
        np.testing.assert_allclose(attn_sum, np.ones((10, 4)), rtol=1e-6)


if __name__ == "__main__":
    main()

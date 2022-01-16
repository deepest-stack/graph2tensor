#!/usr/bin/env python3

from graph2tensor.model.layers import RGCNConv
from unittest import TestCase, main
import numpy as np
import tensorflow as tf


class TestRGCN(TestCase):

    def test_config(self):
        rgcn = RGCNConv(input_dims=128, num_relations=100, units=32, regularizing='block', num_basis_block=8, name="rgcn")
        config = rgcn.get_config()
        assert config["input_dims"] == 128
        assert config["num_relations"] == 100
        assert config["units"] == 32
        assert config["regularizing"] == 'block'
        assert config["num_basis_block"] == 8
        assert config["name"] == "rgcn"
        _ = RGCNConv.from_config(config)

    def test_regularizing(self):
        with self.assertRaises(Warning):
            _ = RGCNConv(input_dims=128, num_relations=100, regularizing='xxx')

    def test_num_basis_block(self):
        with self.assertRaises(ValueError):
            _ = RGCNConv(input_dims=128, num_relations=100, units=32,
                         regularizing='block', num_basis_block=12, name="rgcn")

    def test_call(self):
        hop = (
            tf.constant(np.random.random(size=(10, 16)), dtype=tf.float64),
            tf.constant(np.random.randint(100, size=55), dtype=tf.int32),
            tf.constant(np.random.random(size=(55, 16)), dtype=tf.float64),
            np.repeat(np.arange(10), np.arange(1, 11))
        )
        rgcn = RGCNConv(input_dims=16, num_relations=100, units=32,
                        regularizing='block', num_basis_block=8, name="rgcn")
        assert rgcn(hop).numpy().shape == (10, 32)
        rgcn = RGCNConv(input_dims=16, num_relations=100, units=32,
                        regularizing='basis', num_basis_block=8, name="rgcn")
        assert rgcn(hop).numpy().shape == (10, 32)


if __name__ == "__main__":
    main()

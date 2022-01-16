#!/usr/bin/env python3
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
from graph2tensor.model.layers import GCNConv
from graph2tensor.model.models import MessagePassing
from unittest import TestCase, main

conv_layers = [
    GCNConv(units=32, name="layer1"),
    GCNConv(units=32, name="layer2"),
    GCNConv(units=8, name="layer3"),
]
mp = MessagePassing([conv_layers, conv_layers, conv_layers], name="sage", concat_hidden=False)
src = {'feat': tf.constant(np.random.random(size=(4, 16)), dtype=tf.float32)}
hop1 = {'feat': tf.constant(np.random.random(size=(10, 16)), dtype=tf.float32)}
hop2 = {'feat': tf.constant(np.random.random(size=(55, 16)), dtype=tf.float32)}
hop3 = {'feat': tf.constant(np.random.random(size=(110, 16)), dtype=tf.float32)}
edge = {}
hops = (
    (src, edge, hop1, tf.repeat(tf.range(4), tf.range(1, 5))),
    (hop1, edge, hop2, tf.repeat(tf.range(10), tf.range(1, 11))),
    (hop2, edge, hop3, tf.repeat(tf.range(55), 2))
)


class TestMsgPassing(TestCase):

    def test_config(self):
        config = mp.get_config()
        assert config["name"] == "sage"
        assert config["concat_hidden"] is False
        assert config["attr_reduce_mode"] == 'concat'
        assert config["conv_layers"].__len__() == 3
        assert config["conv_layers"][0].__len__() == 3
        assert config["conv_layers"][0][1]["class_name"] == "GCNConv"
        assert config["conv_layers"][0][1]["config"]["name"] == "layer2"
        assert config["conv_layers"][0][1]["config"]["units"] == 32
        custom_objects = {
            "GCNConv": GCNConv,
        }
        _ = MessagePassing.from_config(config, custom_objects)

    def test_save_load(self):
        x = mp((hops, hops, hops)).numpy()
        mp.save_weights("/tmp/sage")
        mp1 = MessagePassing([conv_layers, conv_layers, conv_layers], name="sage", concat_hidden=False)
        mp1.load_weights("/tmp/sage")
        x1 = mp1((hops, hops, hops)).numpy()
        np.testing.assert_allclose(x, x1, atol=1e-6)

    def test_call(self):
        assert mp((hops, hops, hops)).numpy().shape == (4, 8)
        mp1 = MessagePassing([conv_layers, conv_layers, conv_layers], name="sage", concat_hidden=True)
        assert mp1((hops, hops, hops)).numpy().shape == (4, 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import tensorflow as tf


class GCNConv(tf.keras.layers.Layer):
    r"""
    GCN convolution layer

    .. math::
        h_i^{(l+1)} = h_i^{(l)} + avg(\mathbf{W} h_j^{(l)} + b, j \in \mathcal{N}_i)

    :param units: number of hidden units
    :param add_self_loop: whether add self loop to each node, default to `True`
    :param activation: layer activation, `str` or `tf.keras.activations` object,
                        default to linear activation
    :param use_bias: whether add bias or not, default to `True`
    :param kwargs: args passed to `tf.keras.layers.Layer`
    """

    def __init__(self, units=32, add_self_loop=True, activation=None, use_bias=True, **kwargs):
        super(GCNConv, self).__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
        self.add_self_loop = add_self_loop

    def get_config(self):
        config = super(GCNConv, self).get_config()
        linear_config = self.linear.get_config()
        config.update(
            {
                "units": linear_config["units"],
                "add_self_loop": self.add_self_loop,
                "activation": linear_config["activation"],
                "use_bias": linear_config["use_bias"]
            }
        )
        return config

    def call(self, inputs, **kwargs):
        src, _, dst, segment_ids = inputs
        x = tf.math.segment_mean(dst, tf.squeeze(segment_ids))
        if self.add_self_loop:
            x = x + src
        return self.linear(x)

    def explain_call(self, inputs):
        src, edge, dst, segment_ids, weights = inputs
        dst =  tf.multiply(dst, tf.reshape(weights, (-1, 1)))
        return self.call((src, edge, dst, segment_ids))


if __name__ == "__main__":
    pass

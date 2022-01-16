#!/usr/bin/env python3

import tensorflow as tf


class GINConv(tf.keras.layers.Layer):
    r"""
    Graph Isomorphism Network layer.

    .. math::
        h_v^{(k)} = MLP^{(k)}((1+\epsilon^{(k)}) \cdot h_v^{(k-1)}
        + aggr(\{h_u^{(k-1)},{u\in \mathcal{N}(v)}\})

    or

    .. math::
        h_v^{(k)} = MLP^{(k)}((1+\epsilon^{(k)}) \cdot h_v^{(k-1)}
        + aggr(\{w_{uv}h_u^{(k-1)},{u\in \mathcal{N}(v)}\})

    if edge weight is available

    :param init_eps: the initial epsilon value, default to 0
    :param aggr_type: how to aggregate message from neighbours, expected "max", "mean"
                      or "sum", default to "max"
    :param mlp_units: `int` or `list` of integer, the units of each layer in mlp, an
                        `int` value will define a 1-layer mlp, default to 32
    :param mlp_activations: `str` or `tf.keras.activations` object, or list of it, the
                            activation of each layer in mlp, default to linear activation
    :param kwargs: args passed to `tf.keras.layers.Layer`
    """

    def __init__(self, init_eps=.0, aggr_type='max', mlp_units=32, mlp_activations=None, **kwargs):
        super(GINConv, self).__init__(**kwargs)
        self.init_eps = init_eps
        self.eps = self.add_weight(
            name="epsilon",
            trainable=True,
            shape=(1,),
            initializer=tf.constant_initializer(init_eps)
        )
        self.aggr_type = aggr_type
        if self.aggr_type.upper() not in ("MAX", "MEAN", "SUM"):
            self.aggr_type = "max"
            raise Warning("Unrecognized `aggr_type`: %s, 'max' aggregation"
                          " will be applied." % (aggr_type, ))

        if not isinstance(mlp_units, list):
            mlp_units = [mlp_units]
        if not isinstance(mlp_activations, list):
            mlp_activations = [mlp_activations]
        if len(mlp_units) != len(mlp_activations):
            if len(mlp_units) == 1:
                mlp_units = mlp_units * len(mlp_activations)
            elif len(mlp_activations) == 1:
                mlp_activations = mlp_activations * len(mlp_units)
            else:
                raise ValueError("`mlp_units` and `mlp_activations` should have same length")
        self.mlp_layers = [
            tf.keras.layers.Dense(units, activation=activation)
            for units, activation in zip(mlp_units, mlp_activations)
        ]

    def get_config(self):
        mlp_units, mlp_activations = [], []
        for layer in self.mlp_layers:
            config = layer.get_config()
            mlp_units.append(config["units"])
            mlp_activations.append(config["activation"])
        config = super(GINConv, self).get_config()
        config.update(
            {
                "init_eps": self.init_eps,
                "mlp_units": mlp_units,
                "mlp_activations": mlp_activations
            }
        )
        return config

    def call(self, inputs, **kwargs):
        src, edge_weight, dst, segment_ids = inputs
        if kwargs.get("edge_weighted", False):
            dst = dst * edge_weight
        if self.aggr_type.upper() == "MAX":
            x = tf.math.segment_max(dst, tf.squeeze(segment_ids))
        elif self.aggr_type.upper() == "MEAN":
            x = tf.math.segment_mean(dst, tf.squeeze(segment_ids))
        else:
            x = tf.math.segment_sum(dst, tf.squeeze(segment_ids))
        x = x + (1 + self.eps) * src
        for layer in self.mlp_layers:
            x = layer(x)
        return x

    def explain_call(self, inputs):
        src, edge, dst, segment_ids, weights = inputs
        dst =  tf.multiply(dst, tf.reshape(weights, (-1, 1)))
        return self.call((src, edge, dst, segment_ids))


if __name__ == "__main__":
    pass

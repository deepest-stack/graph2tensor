#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import activations


class GATConv(tf.keras.layers.Layer):
    r"""
    Graph Attention Convolution layer.

    .. math::
        \overrightarrow{h_i^{\prime}} =
        \sigma (\sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij} \mathbf{W} \overrightarrow{h_j})

    where

    .. math::
        \alpha_{ij} = \frac{exp(LeakyReLU(\overrightarrow{\mathbf{a}}^T
        [\mathbf{W}\overrightarrow{h_i}||\mathbf{W}\overrightarrow{h_j}]))}
        {\sum_{k\in\mathcal{N}_i\cup\{i\}}exp(LeakyReLU(\overrightarrow{\mathbf{a}}^T
        [\mathbf{W}\overrightarrow{h_i}||\mathbf{W}\overrightarrow{h_k}]))}

    For multi-heads attention, when "mean" reduction is applied,

    .. math::
        \overrightarrow{h_i^{\prime}} =
        \sigma (\frac{1}{K}\sum_{k=1}^K\sum_{j \in \mathcal{N}_i \cup \{i\}}
        \alpha_{ij}^k \mathbf{W}^k \overrightarrow{h_j})

    which activate after averaging, and

    .. math::
        \overrightarrow{h_i^{\prime}} =
        \mathop{||}\limits_{k=1}^{K} \sigma (\sum_{j \in \mathcal{N}_i \cup \{i\}}
        \alpha_{ij}^k \mathbf{W}^k \overrightarrow{h_j})

    when "concat" reduction is applied, which concatenate after activation.

    :param units: number of hidden units
    :param num_heads: number of attention headers, default to 1
    :param negative_slope: negative slope of the `leaky_relu` activation to attention,
                            default to 0.2
    :param activation: layer activation, `str` or `tf.keras.activations` object,
                        default to linear activation
    :param bias: whether add bias or not, default to `True`
    :param multi_heads_reduction: how to reduce the outputs of multiple headers,
                                  expect "concat" or "mean", default to "mean"
    :param kwargs: args passed to `tf.keras.layers.Layer`
    """

    def __init__(self,
                 units=32,
                 num_heads=1,
                 negative_slope=0.2,
                 activation=None,
                 bias=True,
                 multi_heads_reduction='concat',
                 **kwargs):
        super(GATConv, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.linear = tf.keras.layers.Dense(
            units=self.units*self.num_heads,
            use_bias=False
        )
        self.negative_slope = negative_slope
        self.src_attn = self.add_weight(
            shape=(1, self.num_heads, self.units),
            initializer="random_normal",
            trainable=True,
            name="src_attr"
        )
        self.dst_attn = self.add_weight(
            shape=(1, self.num_heads, self.units),
            initializer="random_normal",
            trainable=True,
            name="dst_attr"
        )
        self.bias = bias
        if self.bias:
            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, name="bias"
            )
        self.activation = activations.get(activation)
        self.multi_heads_reduction = multi_heads_reduction
        if self.multi_heads_reduction not in ('concat', 'mean'):
            self.multi_heads_reduction = 'concat'
            raise Warning("Unrecognized `multi_heads_reduction` mode: %s, concat reduction"
                          " will be applied." % self.multi_heads_reduction)

    def get_config(self):
        config = super(GATConv, self).get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "negative_slope": self.negative_slope,
                "activation": activations.serialize(self.activation),
                "bias": self.bias,
                "multi_heads_reduction": self.multi_heads_reduction
            }
        )
        return config

    def call(self, inputs, **kwargs):
        src, _, dst, segment_ids = inputs
        segment_ids = tf.squeeze(segment_ids)
        h, c = self.num_heads, self.units
        x_src, x_dst = tf.reshape(self.linear(src), (-1, h, c)), \
                       tf.reshape(self.linear(dst), (-1, h, c))
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        a_l = tf.multiply(x_src, self.src_attn)
        a_r = tf.multiply(x_dst, self.dst_attn)
        attn = tf.reduce_sum(
            tf.gather(a_l, segment_ids) + a_r,
            axis=-1,
            keepdims=False
        )
        # add "self attention"
        self_attn = tf.reduce_sum(
            tf.multiply(x_src, self.src_attn) + tf.multiply(x_src, self.dst_attn),
            axis=-1,
            keepdims=False
        )
        attn = tf.exp(tf.nn.leaky_relu(attn, alpha=self.negative_slope))
        self_attn = tf.exp(tf.nn.leaky_relu(self_attn, alpha=self.negative_slope))
        attn_norm = tf.math.segment_sum(attn, segment_ids) + self_attn
        attn = tf.divide(attn, tf.gather(attn_norm, segment_ids))
        self_attn = tf.divide(self_attn, attn_norm, segment_ids)
        x1 = tf.multiply(x_dst, tf.expand_dims(attn, axis=-1))
        x1 = tf.math.segment_sum(x1, segment_ids)
        x2 = tf.multiply(x_src, tf.expand_dims(self_attn, axis=-1))
        x = x1 + x2
        if self.multi_heads_reduction == 'mean':
            x = self.activation(tf.reduce_mean(x, axis=1))
        else:
            x = tf.reshape(self.activation(x), (-1, h * c))
        if kwargs.get("with_attention"):
            return x, tf.concat((attn, self_attn), axis=0)
        else:
            return x

    def explain_call(self, inputs):
        src, edge, dst, segment_ids, weights = inputs
        dst =  tf.multiply(dst, tf.reshape(weights, (-1, 1)))
        return self.call((src, edge, dst, segment_ids))

if __name__ == "__main__":
    pass

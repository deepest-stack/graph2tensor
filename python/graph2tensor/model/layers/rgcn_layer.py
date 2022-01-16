#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import activations
import numpy as np


class RGCNConv(tf.keras.layers.Layer):
    r"""
    Relation GCN convolution layer.

    .. math::
        h_i^{(l+1)} = \sigma (\sum_{r \in \mathcal{R}}
        \sum_{j \in \mathcal{N}_i^r} \frac{1}{|\mathcal{N}_i|}
        \mathbf{W}_r^{(l)}h_j^{(l)} + \mathbf{W}_0^{(l)}h_i^{(l)})

    For basis regularizing,

    .. math::
        \mathbf{W}_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    and for block regularizing

    .. math::
        \mathbf{W}_r^{(l)} = \bigoplus_{b=1}^{B}Q_{br}^{(l)}


    where :math:`\bigoplus` is block-diagonal-composition operator.

    :param input_dims: input dimension
    :param num_relations: number of the relation type
    :param units: number of hidden units
    :param regularizing: how to regularize the relation embedding,
                        expected "basis" or "block", default to "basis"
    :param num_basis_block: the number of basis vectors or number of blocks,
                            for "block" regularizing, `num_basis_block` should
                            be divider of `input_dims` and `units`
    :param activation: layer activation, `str` or `tf.keras.activations` object,
                        default to linear activation
    :param bias: whether add bias or not, default to `True`
    :param kwargs: args passed to `tf.keras.layers.Layer`

    .. note::
        This layer suppose each edge in ego-graph must have one and only one
        attribute, which is the type of edge(relation).
    """

    def __init__(self,
                 input_dims,
                 num_relations,
                 units=32,
                 regularizing='basis',
                 num_basis_block=4,
                 activation=None,
                 bias=True,
                 **kwargs):
        super(RGCNConv, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.num_relations = num_relations
        self.units = units
        self.regularizing = regularizing
        self.num_basis_block = num_basis_block
        if self.regularizing not in ('basis', 'block'):
            self.regularizing = 'basis'
            raise Warning("Unrecognized regularizing mode: %s"
                          ", basis regularizing will be applied" % self.regularizing)
        if self.regularizing == 'basis':
            self.w = self.add_weight(
                shape=(self.num_basis_block, self.input_dims, self.units),
                initializer="random_normal",
                trainable=True,
                name="w"
            )
            self.coefficients = self.add_weight(
                shape=(self.num_relations, self.num_basis_block, 1, 1),
                initializer="random_normal",
                trainable=True,
                name="coefficients"
            )
        else:
            if self.input_dims % self.num_basis_block != 0 or self.units % self.num_basis_block != 0:
                raise ValueError("num_basis_block should be divider of input_dims and units")
            self.w = self.add_weight(
                shape=(self.num_relations, self.input_dims, self.units),
                initializer="random_normal",
                trainable=True,
                name="w"
            )
            mask = np.zeros((self.input_dims, self.units), dtype=np.float32)
            row_step, col_step = self.input_dims//self.num_basis_block, self.units//self.num_basis_block
            for i in range(self.num_basis_block):
                mask[i*row_step:(i+1)*row_step, i*col_step:(i+1)*col_step] = 1.
            self.mask = tf.expand_dims(tf.constant(mask), axis=0)
        self.w0 = self.add_weight(
            shape=(self.input_dims, self.units),
            initializer="random_normal",
            trainable=True,
            name="w0"
        )
        self.bias = bias
        if self.bias:
            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, name="bias"
            )
        self.activation = activations.get(activation)

    def get_config(self):
        config = super(RGCNConv, self).get_config()
        config.update(
            {
                "input_dims": self.input_dims,
                "num_relations": self.num_relations,
                "units": self.units,
                "regularizing": self.regularizing,
                "num_basis_block": self.num_basis_block,
                "activation": activations.serialize(self.activation),
                "bias": self.bias
            }
        )
        return config

    def call(self, inputs, **kwargs):
        src, etype, dst, segment_ids = inputs
        if self.regularizing == 'basis':
            coef = tf.nn.embedding_lookup(self.coefficients, tf.squeeze(etype))
            w = tf.reduce_sum(
                tf.multiply(coef, self.w),
                axis=1
            )
        else:
            w = tf.nn.embedding_lookup(tf.multiply(self.w, self.mask), tf.squeeze(etype))
        x = tf.reduce_sum(
            tf.multiply(tf.expand_dims(dst, axis=-1), w),
            axis=1
        )
        x = tf.math.segment_mean(x, tf.squeeze(segment_ids))
        x = x + tf.matmul(src, self.w0)
        if self.bias:
            x = x + self.b
        return self.activation(x)

    def explain_call(self, inputs):
        src, edge, dst, segment_ids, weights = inputs
        dst =  tf.multiply(dst, tf.reshape(weights, (-1, 1)))
        return self.call((src, edge, dst, segment_ids))


if __name__ == "__main__":
    pass

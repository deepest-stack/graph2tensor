#!/usr/bin/env python3

import tensorflow as tf


class UniMP(tf.keras.layers.Layer):
    r"""
    `Unified Message Passing Model <https://arxiv.org/pdf/2009.03509v5.pdf>`__.

    .. math::
        \begin{aligned}
        q_{c,i}^{(l)} &= W_{c,q}^{(l)}h_{i}^{(l)} + b_{c,q}^{(l)} \\
        k_{c,j}^{(l)} &= W_{c,k}^{(l)} + b_{c,k}^{(l)} \\
        e_{c,ij} &= W_{c,e}e_{ij} + b_{c,e} \\
        \alpha_{c,ij}^{(l)} &= \frac {exp(q_{c,i}^{(l)}
        \cdot (k_{c,j}^{(l)} + e_{c,ij}) / \sqrt{d})}{\sum_{u \in
        \mathcal{N}(i)}exp(q_{c,i}^{(l)} \cdot (k_{c,u}^{(l)} + e_{c,iu}) / \sqrt{d})} \\
        v_{c,j}^{(l)} &= W_{c,v}^{(l)}h_j^{(l)} + b_{c,v}^{(l)}
        \end{aligned}

    where :math:`d` is the number of hidden units.

    For multi-heads attention, when "mean" reduction is applied,

    .. math::
        \hat{h}_{i}^{(l+1)} = avg(\sum_{j \in \mathcal{N}(i)}
        \alpha_{c,ij}^{(l)}(v_{c,j}^{(l)} + e_{c,ij}))

    and when "concat" reduction is applied,

    .. math::
        \hat{h}_{i}^{(l+1)} = \mathop{||}\limits_{c=1}^{C}
        (\sum_{j \in \mathcal{N}(i)} \alpha_{c,ij}^{(l)}(v_{c,j}^{(l)} + e_{c,ij}))

    Add gated residual connection to prevent model from over-smothing,

    .. math::
        \begin{aligned}
        r_{i}^{(l)} &= W_{r}^{l}h_{i}^{(l)} + b_{r}^{(l)} \\
        \beta_{i}^{(l)} &= sigmoid(W_{g}^{(l)}[\hat{h}_{i}^{(l+1)}
        \mathop{||} r_{i}^{(l)} \mathop{||} (\hat{h}_{i}^{(l+1)}-r_{i}^{(l)})]) \\
        h_{i}^{(l+1)} &= (1 - \beta_{i}^{l})\hat{h}_{i}^{(l+1)} + \beta_{i}^{l}r_{i}^{(l)}
        \end{aligned}

    For "concat" reduction, layer normalization and relu activation will
    also be applied,

    .. math::
        h_{i}^{(l+1)} = ReLU(LayerNorm((1 - \beta_{i}^{l})\hat{h}_{i}^{(l+1)}
        + \beta_{i}^{l}r_{i}^{(l)}))

    :param units: number of hidden units
    :param num_heads: number of attention headers, default to 1
    :param multi_heads_reduction: how to reduce the outputs of multiple headers,
                                  expect "concat" or "mean", default to "mean"
    :param kwargs: args passed to `tf.keras.layers.Layer`
    """

    def __init__(self,
                 units=32,
                 num_heads=1,
                 multi_heads_reduction='concat',
                 **kwargs):
        super(UniMP, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.query = tf.keras.layers.Dense(units=num_heads*units)
        self.dst_key = tf.keras.layers.Dense(units=num_heads*units)
        self.edge_key = tf.keras.layers.Dense(units=num_heads*units)
        self.multi_heads_reduction = multi_heads_reduction
        if self.multi_heads_reduction not in ('concat', 'mean'):
            self.multi_heads_reduction = 'concat'
            raise Warning("Unrecognized `multi_heads_reduction` mode: %s, concat reduction"
                          " will be applied." % self.multi_heads_reduction)
        self.linear = tf.keras.layers.Dense(units=units)
        if self.multi_heads_reduction == "concat":
            self.res_linear = tf.keras.layers.Dense(units=num_heads*units)
            self.layer_norm = tf.keras.layers.LayerNormalization(scale=False)
        else:
            self.res_linear = tf.keras.layers.Dense(units=units)
        self.beta = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def get_config(self):
        config = super(UniMP, self).get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "multi_heads_reduction": self.multi_heads_reduction
            }
        )
        return config

    def call(self, inputs, **kwargs):
        src, edata, dst, segment_ids = inputs
        segment_ids = tf.squeeze(segment_ids)
        h, c = self.num_heads, self.units
        query, key_dst, key_edge = tf.reshape(self.query(src), (-1, h, c)), \
                               tf.reshape(self.dst_key(dst), (-1, h, c)), \
                               tf.reshape(self.edge_key(edata), (-1, h, c))

        attn = tf.reduce_sum(
            tf.multiply(tf.gather(query, segment_ids), key_dst+key_edge),
            axis=-1
        ) / tf.sqrt(1.0*self.units)
        attn = tf.exp(attn)
        attn_norm = tf.gather(tf.math.segment_sum(attn, segment_ids), segment_ids)
        attn = tf.divide(attn, attn_norm)
        x = tf.multiply(tf.expand_dims(self.linear(dst), axis=1), tf.expand_dims(attn, axis=-1))
        x = tf.math.segment_sum(x, segment_ids)
        if self.multi_heads_reduction == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = tf.reshape(x, (-1, h*c))
        # gated skip connect
        res = self.res_linear(src)
        beta = self.beta(tf.concat((x, res, x-res), axis=1))
        x = (1-beta) * x + beta * res
        if self.multi_heads_reduction == 'concat':
            x = tf.nn.relu(self.layer_norm(x))
        if kwargs.get("with_attention"):
            return x, attn
        else:
            return x

    def explain_call(self, inputs):
        src, edge, dst, segment_ids, weights = inputs
        dst =  tf.multiply(dst, tf.reshape(weights, (-1, 1)))
        return self.call((src, edge, dst, segment_ids))


if __name__ == "__main__":
    pass

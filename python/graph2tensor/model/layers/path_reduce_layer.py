#!/usr/bin/env python3

import tensorflow as tf


class MeanPathReduce(tf.keras.layers.Layer):
    r"""
    Apply mean reduction on outputs of each path.
    """

    def call(self, inputs, training=None):
        return tf.reduce_mean(inputs, axis=0)

class MaxPathReduce(tf.keras.layers.Layer):
    r"""
    Apply max reduction on outputs of each path.
    """

    def call(self, inputs, training=None):
        return tf.reduce_max(inputs, axis=0)

class SumPathReduce(tf.keras.layers.Layer):
    r"""
    Apply sum reduction on outputs of each path.
    """

    def call(self, inputs, training=None):
        return tf.reduce_sum(inputs, axis=0)

class ConcatPathReduce(tf.keras.layers.Layer):
    r"""
    Apply concatenate reduction on outputs of each path.
    """

    def call(self, inputs, training=None):
        return tf.concat(inputs, axis=1)
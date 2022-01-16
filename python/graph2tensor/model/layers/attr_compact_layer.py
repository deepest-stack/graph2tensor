#!/usr/bin/env python3
import tensorflow as tf
from ...converter.ego2tensor import NID, EID, UID, VID

reserved_attrs = {NID, EID, UID, VID}


class AttrCompact(tf.keras.layers.Layer):

    """
    Pre-processing layers that transform nodes & edges attributes from `dict` of tensor
    into one compact tensor. Usually used as the last pre-processing layer.

    :param mode: how to compact attributes, expected 'concat', 'mean' or 'sum', default to 'concat'
    :param name: name of layer
    :param kwargs: args passed to `tf.keras.layers.Layer`

    :Example:
    >>> concat = AttrCompact(mode='concat', name="concat")
    """

    def __init__(self, mode='concat', name=None, **kwargs):
        super(AttrCompact, self).__init__(name=name, **kwargs)
        self.mode = mode

    def get_config(self):
        config = super(AttrCompact, self).get_config()
        config["mode"] = self.mode
        return config

    def compact(self, attrs):
        attrs_ = {}
        for k in attrs:
            if k not in reserved_attrs: attrs_[k] = attrs[k]
        if attrs_:
            if self.mode == 'mean':
                return tf.reduce_mean(
                    [tf.cast(v, tf.float32) for v in attrs_.values()],
                    axis=0
                )
            elif self.mode == 'sum':
                return tf.reduce_sum(
                    [tf.cast(v, tf.float32) for v in attrs_.values()],
                    axis=0
                )
            else:
                keys = sorted(attrs_.keys())
                if len(keys) > 1:
                    return tf.concat(
                        [tf.cast(attrs_[k], tf.float32) for k in keys],
                        axis=1
                    )
                else:
                    return tf.cast(attrs_[keys[0]], tf.float32)

        else:
            return tf.constant([], dtype=tf.float32)

    def call(self, inputs, **kwargs):
        out = []
        for path in inputs:
            path_compact = []
            for src, edge, dst, segment_ids in path:
                src_compact = self.compact(src)
                edge_compact = self.compact(edge)
                dst_compact = self.compact(dst)
                path_compact.append((src_compact, edge_compact, dst_compact, segment_ids))
            out.append(tuple(path_compact))
        return tuple(out)

    def explain_call(self, inputs, **kwargs):
        out = []
        for path in inputs:
            path_compact = []
            for src, edge, dst, segment_ids, weights in path:
                src_compact = self.compact(src)
                edge_compact = self.compact(edge)
                dst_compact = self.compact(dst)
                path_compact.append((src_compact, edge_compact, dst_compact, segment_ids, weights))
            out.append(tuple(path_compact))
        return tuple(out)
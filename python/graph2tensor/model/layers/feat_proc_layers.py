#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Embedding as TFEmbedding
from tensorflow.keras.layers.experimental.preprocessing \
    import CategoryEncoding, StringLookup, IntegerLookup
from copy import deepcopy


class FeatureEncoder(tf.keras.layers.Layer):

    def __init__(self, name, cate_feat_def, **kwargs):
        super(FeatureEncoder, self).__init__(name=name, **kwargs)
        self._cate_feat_def = cate_feat_def
        self._encoders = {}

    def get_config(self):
        config = super(FeatureEncoder, self).get_config()
        config["cate_feat_def"] = self._cate_feat_def
        return config

    def call(self, inputs, **kwargs):
        out = []
        for path in inputs:
            path_concat = []
            for src, edge, dst, offset in path:
                src_emb = {k: self._encoders[k](v) if k in self._encoders else v
                           for k, v in src.items()}
                edge_emb = {k: self._encoders[k](v) if k in self._encoders else v
                                for k, v in edge.items()}
                dst_emb = {k: self._encoders[k](v) if k in self._encoders else v
                           for k, v in dst.items()}
                path_concat.append((src_emb, edge_emb, dst_emb, offset))
            out.append(tuple(path_concat))
        return tuple(out)

    def explain_call(self, inputs, **kwargs):
        out = []
        for path in inputs:
            path_concat = []
            for src, edge, dst, offset, weights in path:
                src_emb = {k: self._encoders[k](v) if k in self._encoders else v
                           for k, v in src.items()}
                edge_emb = {k: self._encoders[k](v) if k in self._encoders else v
                                for k, v in edge.items()}
                dst_emb = {k: self._encoders[k](v) if k in self._encoders else v
                           for k, v in dst.items()}
                path_concat.append((src_emb, edge_emb, dst_emb, offset, weights))
            out.append(tuple(path_concat))
        return tuple(out)


class Embedding(TFEmbedding):

    def __init__(self, **kwargs):
        super(Embedding, self).__init__(**kwargs)

    def call(self, inputs):
        # x = tf.squeeze(inputs)
        return super(Embedding, self).call(tf.reshape(inputs, (-1, )))


class EmbeddingEncoder(FeatureEncoder):
    """
    Embedding lookup layer with reduction. This layer will not change the `dict` type
    of nodes & edges attributes

    :param name: name of layer
    :param cate_feat_def: sequence of `dict` which has a required key `attr_name` sepcifying
                            the attribute that will be applied embedding lookup, and other key-value
                            args required by `tk.keras.layers.Embedding`. Only the attributes
                            whose name match the keys in `cate_feat_def` will be processed by
                            the corresponding encoder.
    :param kwargs: args passed to `tf.keras.layers.Layer`

    :Example:
    >>> emb = EmbeddingEncoder(
    ...     "embedding_encoder",
    ...     [{"attr_name": "str_feat", "input_dim": 4, "output_dim": 32},
    ...      {"attr_name": "int_feat", "input_dim": 10, "output_dim": 32},]
    ... )
    """

    def __init__(self, name, cate_feat_def, **kwargs):
        super(EmbeddingEncoder, self).__init__(name, cate_feat_def, **kwargs)
        for encoder_def in deepcopy(self._cate_feat_def):
            attr_name = encoder_def.pop("attr_name")
            self._encoders[attr_name] = Embedding(**encoder_def)


class OnehotEncoder(FeatureEncoder):
    """
    Onehot encoding layer. This layer will not change the `dict` type
    of nodes & edges attributes

    :param name: name of layer
    :param cate_feat_def: sequence of `dict` which has a required key `attr_name` sepcifying
                            the attribute that will be applied onehot encoding, and other key-value
                            args required by `tk.keras.layers.experimental.preprocessing.CategoryEncoding`.
                            Only the attributes whose name match the keys in `cate_feat_def`
                            will be processed by the corresponding encoder.
    :param kwargs: args passed to `tf.keras.layers.Layer`

    :Example:
    >>> onehot = OnehotEncoder(
    ...     "onehot_encoder",
    ...     [{"attr_name": "int_feat", "num_tokens": 10}, ]
    ... )
    """

    def __init__(self, name, cate_feat_def, **kwargs):
        super(OnehotEncoder, self).__init__(name, cate_feat_def, **kwargs)
        for encoder_def in deepcopy(self._cate_feat_def):
            attr_name = encoder_def.pop("attr_name")
            self._encoders[attr_name] = CategoryEncoding(**encoder_def)


class StringLookupEncoder(FeatureEncoder):
    """
    String lookup layer. This layer will not change the `dict` type
    of nodes & edges attributes

    :param name: name of layer
    :param cate_feat_def: sequence of `dict` which has a required key `attr_name` sepcifying
                            the attribute that will be applied string lookup, and other key-value
                            args required by `tk.keras.layers.experimental.preprocessing.StringLookup`.
                            Only the attributes whose name match the keys in `cate_feat_def`
                            will be processed by the corresponding encoder.
    :param kwargs: args passed to `tf.keras.layers.Layer`

    :Example:
    >>> string_lookup = StringLookupEncoder(
    ...     "string_lookup",
    ...     [{"attr_name": "str_feat", "vocabulary": ["a", "b", "c"],},]
    ... )
    """

    def __init__(self, name, cate_feat_def, **kwargs):
        super(StringLookupEncoder, self).__init__(name, cate_feat_def, **kwargs)
        for encoder_def in deepcopy(self._cate_feat_def):
            attr_name = encoder_def.pop("attr_name")
            self._encoders[attr_name] = StringLookup(**encoder_def)


class IntegerLookupEncoder(FeatureEncoder):
    """
    Integer lookup layer. This layer will not change the `dict` type
    of nodes & edges attributes

    :param name: name of layer
    :param cate_feat_def: sequence of `dict` which has a required key `attr_name` sepcifying
                            the attribute that will be applied integer lookup, and other key-value
                            args required by `tk.keras.layers.experimental.preprocessing.IntegerLookup`.
                            Only the attributes whose name match the keys in `cate_feat_def`
                            will be processed by the corresponding encoder.
    :param kwargs: args passed to `tf.keras.layers.Layer`

    :Example:
    >>> integer_lookup = IntegerLookupEncoder(
    ...     "integer_lookup",
    ...     [{"attr_name": "int_feat", "vocabulary": list(range(1, 10)), "oov_token": 0,}, ]
    ... )
    """

    def __init__(self, name, cate_feat_def, **kwargs):
        super(IntegerLookupEncoder, self).__init__(name, cate_feat_def, **kwargs)
        for encoder_def in deepcopy(self._cate_feat_def):
            attr_name = encoder_def.pop("attr_name")
            self._encoders[attr_name] = IntegerLookup(**encoder_def)


if __name__ == "__main__":
    pass

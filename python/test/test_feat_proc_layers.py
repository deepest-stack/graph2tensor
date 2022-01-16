#!/usr/bin/env python3

from graph2tensor.model.layers import EmbeddingEncoder, OnehotEncoder,\
    StringLookupEncoder, IntegerLookupEncoder
from graph2tensor.model.layers.attr_compact_layer import AttrCompact
from unittest import TestCase, main
import numpy as np
import tensorflow as tf

hop1 = (
    # src feat
    {
        "int_feat": tf.constant(np.random.randint(1, 10, size=(16, 1)), dtype=tf.int32),
        "float_feat": tf.constant(np.random.rand(16, 8), dtype=tf.float32),
        "str_feat": tf.constant(np.random.choice(['a', 'b', 'c'], size=(16, 1)), dtype=tf.string)
    },
    # edge feat
    {},
    # dst feat
    {
        "int_feat": tf.constant(np.random.randint(1, 10, size=(64, 1)), dtype=tf.int32),
        "float_feat": tf.constant(np.random.rand(64, 8), dtype=tf.float32),
        "str_feat": tf.constant(np.random.choice(['a', 'b', 'c'], size=(64, 1)), dtype=tf.string)
    },
    # offset
    tf.constant(np.ones(16)*4, dtype=tf.int32),
)

hop2 = (
    # src feat
    hop1[-2],
    # edge feat
    {},
    # dst feat
    {
        "int_feat": tf.constant(np.random.randint(1, 10, size=(256, 1)), dtype=tf.int32),
        "float_feat": tf.constant(np.random.rand(256, 8), dtype=tf.float32),
        "str_feat": tf.constant(np.random.choice(['a', 'b', 'c'], size=(256, 1)), dtype=tf.string)
    },
    # offset
    tf.constant(np.ones(64)*4, dtype=tf.int32),
)

x = ((hop1, hop2),)


class TestFeatProc(TestCase):

    def test_attr_compact_config(self):
        attr_compact = AttrCompact(mode='mean', name="attr_compact")
        config = attr_compact.get_config()
        assert config["name"] == "attr_compact"
        assert config['mode'] == 'mean'
        _ = AttrCompact.from_config(config)

    def test_attr_compact_call(self):
        attr_compact = AttrCompact(name="attr_compact")
        string_lookup = StringLookupEncoder(
            "string_lookup",
            [{"attr_name": "str_feat", "vocabulary": ["a", "b", "c"], "num_oov_indices": 0},]
        )
        _ = attr_compact(string_lookup(x))
        emb = EmbeddingEncoder(
            "embedding_encoder",
            [{"attr_name": "str_feat", "input_dim": 4, "output_dim": 8},
             {"attr_name": "int_feat", "input_dim": 10, "output_dim": 8},]
        )
        sum_ = AttrCompact(mode='sum', name="attr_compact")
        _ = sum_(emb(string_lookup(x)))
        mean_ = AttrCompact(mode='mean')
        _ = mean_(emb(string_lookup(x)))

    def test_embedding_config(self):
        emb = EmbeddingEncoder(
            "embedding_encoder",
            [{"attr_name": "str_feat", "input_dim": 4, "output_dim": 32},
             {"attr_name": "int_feat", "input_dim": 10, "output_dim": 32},]
        )
        config = emb.get_config()
        assert config["name"] == "embedding_encoder"
        _ = EmbeddingEncoder.from_config(config)

    def test_embedding_call(self):
        emb = EmbeddingEncoder(
            "embedding_encoder",
            [{"attr_name": "str_feat", "input_dim": 4, "output_dim": 32},
             {"attr_name": "int_feat", "input_dim": 10, "output_dim": 32},]
        )
        string_lookup = StringLookupEncoder(
            "string_lookup",
            [{"attr_name": "str_feat", "vocabulary": ["a", "b", "c"], "num_oov_indices": 0},]
        )
        _ = emb(string_lookup(x))

    def test_onehot_config(self):
        onehot = OnehotEncoder(
            "onehot_encoder",
            [{"attr_name": "int_feat", "num_tokens": 10}, ]
        )
        config = onehot.get_config()
        assert config["name"] == "onehot_encoder"
        _ = OnehotEncoder.from_config(config)

    def test_onehot_call(self):
        onehot = OnehotEncoder(
            "onehot_encoder",
            [{"attr_name": "int_feat", "num_tokens": 10}, ]
        )
        _ = onehot(x)

    def test_integerlookup_config(self):
        integer_lookup = IntegerLookupEncoder(
            "integer_lookup",
            [{'attr_name': "int_feat", "vocabulary": list(range(1, 10))}, ]
        )
        config = integer_lookup.get_config()
        assert config["name"] == "integer_lookup"
        _ = IntegerLookupEncoder.from_config(config)

    def test_integerlookup_call(self):
        integer_lookup = IntegerLookupEncoder(
            "integer_lookup",
            [{'attr_name': "int_feat", "vocabulary": list(range(1, 10)), "oov_token": 0,}, ]
        )
        _ = integer_lookup(x)

    def test_stringlookup_config(self):
        string_lookup = StringLookupEncoder(
            "string_lookup",
            [{"attr_name": "str_feat", "vocabulary": ["a", "b", "c"],},]
        )
        config = string_lookup.get_config()
        assert config["name"] == "string_lookup"
        _ = StringLookupEncoder.from_config(config)

    def test_stringlookup_call(self):
        string_lookup = StringLookupEncoder(
            "string_lookup",
            [{"attr_name": "str_feat", "vocabulary": ["a", "b", "c"],},]
        )
        _ = string_lookup(x)


if __name__ == "__main__":
    main()

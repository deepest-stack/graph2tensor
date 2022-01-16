#!/usr/bin/env python3

from unittest import TestCase, main
from graph2tensor.model.data import EgoTensorGenerator, build_output_signature
from graph2tensor.model.data import build_sampling_table
from graph2tensor.model.data import SkipGramGenerator4DeepWalk
from graph2tensor.model.data import SkipGramGenerator4Node2Vec
from graph2tensor.model.data import SkipGramGenerator4MetaPath2Vec
from graph2tensor.client import NumpyGraph
import numpy as np
import tensorflow as tf
from test_utils import graph_setup

g = NumpyGraph()


class TestDataset(TestCase):

    def test_1_setup(self):
        graph_setup(g)

    def test_egotensor_generator(self):
        ids = np.arange(169343)
        labels = np.random.randint(2, size=ids.shape[0], dtype=np.int32)
        output_signature = build_output_signature(g.schema, ["(paper) -[cites]- (paper) -[cites]- (paper)"], False)
        with EgoTensorGenerator(
                graph=g,
                meta_paths=["(paper) -[cites]- (paper) -[cites]- (paper)"],
                sampler_process_num=1,
                converter_process_num=1,
                expand_factors=2,
                strategies="random",
                include_edge=False) as data_gen:
            ds = tf.data.Dataset.from_generator(
                data_gen,
                args=(ids, 1024, False, labels),
                output_signature=output_signature
            ).repeat(2)
            for _ in ds:
                pass

    def test_skipgram_generator_deepwalk(self):
        ids = np.arange(169343)
        sampling_table = build_sampling_table(g, 'cites')
        with SkipGramGenerator4DeepWalk(
                graph=g,
                edge_type='cites',
                vocabulary_size=169343,
                negative_samples=4,
                sampling_table=sampling_table) as data_gen:
            ds = tf.data.Dataset.from_generator(
                data_gen,
                args=(ids, 1024),
                output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64), tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
                                  tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
            ).repeat(2)
            for _ in ds:
                pass

    def test_skipgram_generator_node2vec(self):
        ids = np.arange(169343)
        sampling_table = build_sampling_table(g, 'cites')
        with SkipGramGenerator4Node2Vec(
                graph=g,
                edge_type='cites',
                vocabulary_size=169343,
                negative_samples=4,
                sampling_table=sampling_table) as data_gen:
            ds = tf.data.Dataset.from_generator(
                data_gen,
                args=(ids, 1024),
                output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64), tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
                                  tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
            ).repeat(2)
            for _ in ds:
                pass

    def test_skipgram_generator_metapath2vec(self):
        ids = np.arange(169343)
        sampling_table = build_sampling_table(g, 'cites')
        with SkipGramGenerator4MetaPath2Vec(
                graph=g,
                meta_path="(paper) -[cites]- (paper) -[cites]- (paper)",
                walk_length=6,
                vocabulary_size=169343,
                negative_samples=4,
                sampling_table=sampling_table) as data_gen:
            ds = tf.data.Dataset.from_generator(
                data_gen,
                args=(ids, 1024),
                output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64), tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
                                  tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
            ).repeat(2)
            for _ in ds:
                pass


if __name__ == "__main__":
    main()

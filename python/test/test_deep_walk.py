#!/usr/bin/env python3

from unittest import TestCase, main
from graph2tensor.model.data import SkipGramGenerator4DeepWalk, build_sampling_table
from graph2tensor.model.models import DeepWalk
from graph2tensor.client import NumpyGraph
from test_utils import graph_setup
import tensorflow as tf
import numpy as np

g = NumpyGraph()
deep_walk = DeepWalk(169343, 64, name='deepwalk')


class TestDeepWalk(TestCase):

    def test_1_setup(self):
        graph_setup(g)

    def test_deep_walk(self):
        sampling_table = build_sampling_table(g, 'cites', )
        ids = np.arange(169343)
        deep_walk.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        )
        with SkipGramGenerator4DeepWalk(
                graph=g,
                edge_type='cites',
                vocabulary_size=169343,
                sampling_table=sampling_table,
                negative_samples=4,
                walk_length=5) as data_gen:
            ds = tf.data.Dataset.from_generator(
                data_gen,
                args=(ids, 40960),
                output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64), tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
                                  tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
            )
            # for x, y in ds:
            #     _ = deep_walk(x)
            deep_walk.fit(ds, epochs=1,)

    def test_z_get_node_embeddings(self):
        emb = deep_walk.get_node_embedding(np.random.randint(169343, size=1024), True)
        emb = deep_walk.get_node_embedding(np.random.randint(169343, size=1024), False)

    def test_z_most_similar(self):
        print(deep_walk.most_similar(1, topn=10))


if __name__ == "__main__":
    main()

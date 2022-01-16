#!/usr/bin/env python3

from unittest import TestCase, main
from graph2tensor.model.data import SkipGramGenerator4MetaPath2Vec, build_sampling_table
from graph2tensor.model.models import MetaPath2Vec
from graph2tensor.client import NumpyGraph
from test_utils import graph_setup
import tensorflow as tf
import numpy as np

g = NumpyGraph()
metapath2vec = MetaPath2Vec(169343, 64, name='metapath2vec')


class TestMetaPath2Vec(TestCase):

    def test_1_setup(self):
        graph_setup(g)

    def test_metapath2vec(self):
        sampling_table = build_sampling_table(g, 'cites', )
        ids = np.arange(169343)
        metapath2vec.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        )
        with SkipGramGenerator4MetaPath2Vec(
                graph=g,
                meta_path="(paper) -[cites]- (paper) -[cites]- (paper)",
                vocabulary_size=169343,
                sampling_table=sampling_table,
                negative_samples=4,
                walk_length=6,
        ) as data_gen:
            ds = tf.data.Dataset.from_generator(
                data_gen,
                args=(ids, 40960),
                output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64), tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
                                  tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
            )
            metapath2vec.fit(ds)

    def test_z_get_node_embeddings(self):
        emb = metapath2vec.get_node_embedding(np.random.randint(169343, size=1024), True)
        emb = metapath2vec.get_node_embedding(np.random.randint(169343, size=1024), False)

    def test_z_most_similar(self):
        print(metapath2vec.most_similar(1, topn=10))


if __name__ == "__main__":
    main()

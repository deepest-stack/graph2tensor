#!/usr/bin/env python3
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity


class _Word2Vec(tf.keras.Model):
    """
    Word2Vec model used to learn graph embedding in DeepWalk & Node2Vec, etc..

    :param vocab_size: total number of nodes
    :param embedding_dim: dimension of nodes embedding
    :param num_ns: number of negative nodes
    :param name: node of model
    """
    def __init__(self, vocab_size, embedding_dim, num_ns=5, name=None):
        super(_Word2Vec, self).__init__(name=name)
        self.target_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name="w2v_embedding")
        self.context_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns+1,
            name="w2v_context"
        )
        self.num_ns = num_ns

    @tf.function
    def call(self, inputs, training=None, mask=None):
        tgt, ctx = inputs
        tgt_emb = tf.squeeze(self.target_embedding(tgt))
        ctx_emb = self.context_embedding(ctx)
        return tf.einsum('be,bce->bc', tgt_emb, ctx_emb)

    def get_config(self):
        config = super(_Word2Vec, self).get_config()
        config["vocab_size"] = self.target_embedding.get_config()['input_dim']
        config["embedding_dim"] = self.target_embedding.get_config()['output_dim']
        config["num_ns"] = self.num_ns
        return config

    def get_node_embedding(self, node_indices, concat_target_context=False):
        """
        Get the embedding of given nodes

        :param node_indices: indices of nodes
        :param concat_target_context: whether or not concatenate the context embeddings,
                                      if not, only target embeddings will be returned.
        :return: the embeddings of the given nodes
        """
        emb = self.target_embedding(node_indices, training=False)
        if concat_target_context:
            emb = tf.concat(
                (emb, self.context_embedding(node_indices, training=False)),
                axis=1
            )
        return tf.stop_gradient(emb)

    def most_similar(self, node_index, topn=10):
        """
        Return most `topn` similar nodes of given node based on
        the cosine similarities of node embeddings.

        :param node_index: `int`, index of node
        :param topn: how many nodes to return
        :return: the most `topn` similar nodes indices and its similarities
        """
        node_emb = self.target_embedding.embeddings[node_index].numpy().reshape(1, -1)
        similarities = cosine_similarity(node_emb, self.target_embedding.embeddings.numpy())[0]
        indices = (-similarities).argsort()
        cnt = 0
        rst = []
        for idx in indices:
            if node_index == idx: continue
            rst.append((idx, similarities[idx]))
            cnt += 1
            if cnt >= topn: break
        return rst


class DeepWalk(_Word2Vec):
    """
    Implementation of `DeepWalk <https://arxiv.org/pdf/1403.6652.pdf>`__

    :param vocab_size: total number of nodes
    :param embedding_dim: dimension of nodes embedding
    :param num_ns: number of negative nodes
    :param name: node of model
    """
    pass


class Node2Vec(_Word2Vec):
    """
    Implementation of `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__

    :param vocab_size: total number of nodes
    :param embedding_dim: dimension of nodes embedding
    :param num_ns: number of negative nodes
    :param name: node of model
    """
    pass


class MetaPath2Vec(_Word2Vec):
    """
    Implementation of `MetaPath2Vec <https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf>`__

    :param vocab_size: total number of nodes
    :param embedding_dim: dimension of nodes embedding
    :param num_ns: number of negative nodes
    :param name: node of model
    """
    pass

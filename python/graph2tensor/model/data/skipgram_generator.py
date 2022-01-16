#!/usr/bin/env python3

from .data_generator import DataGenerator


class SkipGramGenerator4DeepWalk(DataGenerator):
    """
    A generator which yield skip-grams.

    :param graph: graph to generate from
    :param edge_type: edge type
    :param vocabulary_size: total number of nodes
    :param walk_length: length to walk
    :param use_edge_probs: whether use edge probabilities as neighbour's distribution,
                           if not or edge probabilities is not found, uniform
                           distribution will be applied
    :param discard_frequent_nodes: whether or not discard frequent nodes
    :param freq_th: the frequency threshold for frequent nodes
    :param window_size: the window size of skip-gram
    :param negative_samples: how many negative nodes to sample for each target node
    :param sampling_table: 1-d `np.ndarray` with length `vocabulary_size`, the
                           distribution for negative sampling, if not specified,
                           uniform distribution will be applied.
    :param sampler_process_num: the number of sampler processes
    :param converter_process_num: the number of converter processes

    :Example:
    >>> with SkipGramGenerator4DeepWalk(
    ...         graph=g,
    ...         edge_type='cites',
    ...         vocabulary_size=169343,
    ...         negative_samples=4,
    ...         sampling_table=sampling_table) as data_gen:
    ...     ds = tf.data.Dataset.from_generator(
    ...         data_gen,
    ...         args=(ids, 40960),
    ...         output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ...                             tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
    ...                           tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
    ...     )
    ...     deepwalk.fit(ds, epochs=5)
    """

    def __init__(self,
                 graph,
                 edge_type,
                 vocabulary_size,
                 walk_length=10,
                 use_edge_probs=False,
                 discard_frequent_nodes=True,
                 freq_th=1e-5,
                 window_size=5,
                 negative_samples=5,
                 sampling_table=None,
                 sampler_process_num=1,
                 converter_process_num=8,
                 **kwargs
                 ):
        sampler_kwargs = {
            "edge_type": edge_type,
            "walk_length": walk_length,
            "use_edge_probs": use_edge_probs,
            "discard_frequent_nodes": discard_frequent_nodes,
            "freq_th": freq_th
        }
        converter_kwargs = {
            "vocabulary_size": vocabulary_size,
            "window_size": window_size,
            "negative_samples": negative_samples,
            "sampling_table": sampling_table
        }
        super(SkipGramGenerator4DeepWalk, self).__init__(
            graph=graph,
            sampler_class='RandomWalker',
            converter_class='SkipGram',
            sampler_kwargs=sampler_kwargs,
            converter_kwargs=converter_kwargs,
            sampler_process_num=sampler_process_num,
            converter_process_num=converter_process_num,
        )


class SkipGramGenerator4Node2Vec(DataGenerator):
    """
    A generator which yield skip-grams.

    :param graph: graph to generate from
    :param edge_type: edge type
    :param vocabulary_size: total number of nodes
    :param walk_length: length to walk
    :param p: return parameter, see `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__
              for more details, only valid for 'Node2VecWalker'
    :param q: in-out parameter, see `Node2Vec <https://arxiv.org/pdf/1607.00653.pdf>`__
              for more details, only valid for 'Node2VecWalker'
    :param use_edge_probs: whether use edge probabilities as neighbour's distribution,
                           if not or edge probabilities is not found, uniform
                           distribution will be applied
    :param discard_frequent_nodes: whether or not discard frequent nodes
    :param freq_th: the frequency threshold for frequent nodes
    :param window_size: the window size of skip-gram
    :param negative_samples: how many negative nodes to sample for each target node
    :param sampling_table: 1-d `np.ndarray` with length `vocabulary_size`, the
                           distribution for negative sampling, if not specified,
                           uniform distribution will be applied.
    :param sampler_process_num: the number of sampler processes
    :param converter_process_num: the number of converter processes

    :Example:
    >>> with SkipGramGenerator4Node2Vec(
    ...         graph=g,
    ...         edge_type='cites',
    ...         vocabulary_size=169343,
    ...         negative_samples=4,
    ...         sampling_table=sampling_table) as data_gen:
    ...     ds = tf.data.Dataset.from_generator(
    ...         data_gen,
    ...         args=(ids, 40960),
    ...         output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ...                             tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
    ...                           tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
    ...     )
    ...     node2vec.fit(ds, epochs=5)
    """

    def __init__(self,
                 graph,
                 edge_type,
                 vocabulary_size,
                 walk_length=10,
                 p=1.0,
                 q=1.0,
                 use_edge_probs=False,
                 discard_frequent_nodes=True,
                 freq_th=1e-5,
                 window_size=5,
                 negative_samples=5,
                 sampling_table=None,
                 sampler_process_num=1,
                 converter_process_num=8,
                 **kwargs
                 ):
        sampler_kwargs = {
            "edge_type": edge_type,
            "walk_length": walk_length,
            "use_edge_probs": use_edge_probs,
            "discard_frequent_nodes": discard_frequent_nodes,
            "freq_th": freq_th,
            'p': p,
            'q': q
        }
        converter_kwargs = {
            "vocabulary_size": vocabulary_size,
            "window_size": window_size,
            "negative_samples": negative_samples,
            "sampling_table": sampling_table
        }
        super(SkipGramGenerator4Node2Vec, self).__init__(
            graph=graph,
            sampler_class='Node2VecWalker',
            converter_class='SkipGram',
            sampler_kwargs=sampler_kwargs,
            converter_kwargs=converter_kwargs,
            sampler_process_num=sampler_process_num,
            converter_process_num=converter_process_num,
        )


class SkipGramGenerator4MetaPath2Vec(DataGenerator):
    """
    A generator which yield skip-grams.

    :param graph: graph to generate from
    :param edge_type: edge type
    :param vocabulary_size: total number of nodes
    :param walk_length: length to walk
    :param window_size: the window size of skip-gram
    :param negative_samples: how many negative nodes to sample for each target node
    :param sampling_table: 1-d `np.ndarray` with length `vocabulary_size`, the
                           distribution for negative sampling, if not specified,
                           uniform distribution will be applied.
    :param sampler_process_num: the number of sampler processes
    :param converter_process_num: the number of converter processes

    :Example:
    >>> with SkipGramGenerator4MetaPath2Vec(
    ...         graph=g,
    ...         edge_type='cites',
    ...         vocabulary_size=169343,
    ...         negative_samples=4,
    ...         sampling_table=sampling_table) as data_gen:
    ...     ds = tf.data.Dataset.from_generator(
    ...         data_gen,
    ...         args=(ids, 40960),
    ...         output_signature=((tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ...                             tf.TensorSpec(shape=(None, 4+1), dtype=tf.int64)),
    ...                           tf.TensorSpec(shape=(None, 4+1), dtype=tf.int32))
    ...     )
    ...     metapath2vec.fit(ds, epochs=5)
    """

    def __init__(self,
                 graph,
                 meta_path,
                 vocabulary_size,
                 walk_length=10,
                 window_size=5,
                 negative_samples=5,
                 sampling_table=None,
                 sampler_process_num=1,
                 converter_process_num=8,
                 **kwargs
                 ):
        sampler_kwargs = {
            "meta_path": meta_path,
            "walk_length": walk_length
        }
        converter_kwargs = {
            "vocabulary_size": vocabulary_size,
            "window_size": window_size,
            "negative_samples": negative_samples,
            "sampling_table": sampling_table
        }
        super(SkipGramGenerator4MetaPath2Vec, self).__init__(
            graph=graph,
            sampler_class='MetaPathRandomWalker',
            converter_class='SkipGram',
            sampler_kwargs=sampler_kwargs,
            converter_kwargs=converter_kwargs,
            sampler_process_num=sampler_process_num,
            converter_process_num=converter_process_num,
        )
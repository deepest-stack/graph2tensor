#!/usr/bin/env python3
from .data_generator import DataGenerator


class EgoTensorGenerator(DataGenerator):
    """
    A callable object returning a generator which yield ego-graph
    that can be feed into tensorflow models.

    :param graph: graph that generate ego-graph from
    :param meta_paths: the meta_paths that sample along,
                       see :class:`graph2tensor.sampler.MetaPathSampler` for details
    :param sampler_process_num: the number of sampler processes
    :param converter_process_num: the number of converter processes
    :param expand_factors: see :class:`graph2tensor.sampler.MetaPathSampler` for details
    :param strategies: see :class:`graph2tensor.sampler.MetaPathSampler` for details
    :param include_edge: see :class:`graph2tensor.converter.Ego2Tensor` for details

    :Example:
    >>> with EgoTensorGenerator(
    ...            graph=g,
    ...            meta_paths=["(paper) -[cite]- (paper) -[cite]- (paper)"],
    ...            sampler_process_num=1,
    ...            converter_process_num=1,
    ...            expand_factors=2,
    ...            strategies="random",
    ...            include_edge=False) as data_gen:
    ...        ds = tf.data.Dataset.from_generator(
    ...            data_gen,
    ...            args=(np.arange(169343), np.random.randint(2, size=169343)),
    ...            output_signature=output_signature
    ...        ).repeat(2)
    """

    def __init__(self,
                 graph,
                 meta_paths,
                 expand_factors=10,
                 strategies="random",
                 include_edge=False,
                 sampler_process_num=1,
                 converter_process_num=8,
                 **kwargs):
        sampler_kwargs = {
            "meta_paths": meta_paths,
            "strategies": strategies,
            "expand_factors": expand_factors
        }
        super(EgoTensorGenerator, self).__init__(
            graph=graph,
            sampler_class='MetaPathSampler',
            converter_class='Ego2Tensor',
            sampler_kwargs=sampler_kwargs,
            converter_kwargs={"include_edge": include_edge},
            sampler_process_num=sampler_process_num,
            converter_process_num=converter_process_num,
        )

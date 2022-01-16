#!/usr/bin/env python3
import traceback
from multiprocessing import Queue, Process
from ... import client, sampler, converter
import numpy as np
from math import ceil
import inspect


class SamplerProcess(Process):

    def __init__(self, name, sampler_class, graph_class, graph_config, seed_queue, sub_g_queue, **kwargs):
        super(SamplerProcess, self).__init__(name=name)
        self._sampler = getattr(sampler, sampler_class)(
            getattr(client, graph_class).from_config(graph_config),
            **kwargs
        )
        self._seed_queue = seed_queue
        self._sub_g_queue = sub_g_queue

    def run(self) -> None:
        while True:
            seeds = self._seed_queue.get()
            labels = None
            if isinstance(seeds, tuple) and seeds.__len__() == 2:
                ids, labels = seeds
            else:
                ids = seeds
            sub_g = self._sampler.sample(ids)
            if labels is None:
                self._sub_g_queue.put(sub_g)
            else:
                self._sub_g_queue.put((sub_g, labels))


class ConverterProcess(Process):

    def __init__(self,
                 name,
                 converter_class,
                 graph_class,
                 graph_config,
                 sub_g_queue,
                 result_queue,
                 **kwargs):
        super(ConverterProcess, self).__init__(name=name)
        cvt = getattr(converter, converter_class)
        kwargs_ = {}
        if 'graph' in inspect.getfullargspec(cvt).args:
            kwargs_['graph'] = getattr(client, graph_class).from_config(graph_config)
        kwargs_.update(kwargs)
        self._converter = cvt(**kwargs_)
        self._sub_g_queue = sub_g_queue
        self._result_queue = result_queue

    def run(self) -> None:
        while True:
            x = self._sub_g_queue.get()
            labels = None
            if isinstance(x, tuple) and x.__len__() == 2:
                sub_g, labels = x
            else:
                sub_g = x
            rst = self._converter.convert(sub_g)
            if labels is None:
                self._result_queue.put(rst)
            else:
                self._result_queue.put((rst, labels))


class DataGenerator(object):

    def __init__(self,
                 graph,
                 sampler_class,
                 converter_class,
                 sampler_kwargs=None,
                 converter_kwargs=None,
                 sampler_process_num=1,
                 converter_process_num=8,
                 **kwargs
                 ):
        self.seed_queue, self.sub_g_queue, self.result_queue = Queue(), Queue(), Queue()
        if not sampler_kwargs:
            sampler_kwargs = {}
        self.sampler_processes = [
            SamplerProcess(
                "sampler_process_%d" % i,
                sampler_class=sampler_class,
                graph_class=graph.__class__.__name__,
                graph_config=graph.to_config(),
                seed_queue=self.seed_queue,
                sub_g_queue=self.sub_g_queue,
                **sampler_kwargs
            )
            for i in range(sampler_process_num)
        ]
        if not converter_kwargs:
            converter_kwargs = {}
        self.converter_processes = [
            ConverterProcess(
                "converter_process_%d" % i,
                converter_class=converter_class,
                graph_class=graph.__class__.__name__,
                graph_config=graph.to_config(),
                sub_g_queue=self.sub_g_queue,
                result_queue=self.result_queue,
                **converter_kwargs
            )
            for i in range(converter_process_num)
        ]

    def __enter__(self):
        for p in self.sampler_processes:
            p.start()
        for p in self.converter_processes:
            p.start()
        return self

    def __call__(self, seed_ids, batch_size=256, shuffle=False, seed_labels=None, **kwargs):
        """
        :param seed_ids: the ids of the seed nodes
        :param batch_size: batch size
        :param shuffle: whether or not shuffle seeds
        :param seed_labels: the labels of the seed nodes
        :return: a generator which yield data produced by sampler & converter
        """
        if seed_labels is not None and len(seed_ids) != len(seed_labels):
            raise ValueError("Ids and labels had unmatched size, with %d ids and %d labels" %
                             (len(seed_ids), len(seed_labels)))
        if shuffle:
            rng = np.random.default_rng()
            idx = np.arange(len(seed_ids))
            rng.shuffle(idx)
            seed_ids = seed_ids[idx]
            if seed_labels is not None:
                seed_labels = seed_labels[idx]

        steps_per_epoch = ceil(seed_ids.shape[0]/batch_size)
        for i in range(steps_per_epoch):
            seeds = seed_ids[i*batch_size:(i+1)*batch_size]
            if seed_labels is not None:
                seeds = (seeds, seed_labels[i*batch_size:(i+1)*batch_size])
            self.seed_queue.put(seeds)
            yield self.result_queue.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.sampler_processes:
            if p.is_alive(): p.terminate()
        for p in self.converter_processes:
            if p.is_alive(): p.terminate()
        self.seed_queue.close()
        self.sub_g_queue.close()
        self.result_queue.close()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
            return False
        return True


if __name__ == "__main__":
    pass

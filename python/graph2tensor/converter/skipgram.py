#!/usr/bin/env python3
import numpy as np
from .base_converter import BaseConverter


class SkipGram(BaseConverter):
    """
    A converter which generate skip-grams (and negative samples) from paths.

    :param vocabulary_size: total numbers of nodes
    :param window_size: the window size of skip-gram
    :param negative_samples: how many negative nodes to sample for each target node
    :param sampling_table: 1-d `np.ndarray` with length `vocabulary_size`, the
                           distribution for negative sampling
    """

    def __init__(self, vocabulary_size, window_size=5, negative_samples=5, sampling_table=None):
        self._vocabulary_size = vocabulary_size
        self._window_size = window_size
        self._negative_samples = max(1, negative_samples)
        self._sampling_table = sampling_table

    def convert(self, paths, **kwargs):
        """
        Generate skip-gram (and negative samples) from paths.

        :param paths: paths to generate skip-gram from
        :return: (`target`, `context`, `labels`), `target` is 1-d `np.ndarray`,
                 `context` is 2-d `np.ndarray` with shape (num_target, negative_samples+1)
                 and `labels` is 2-d `np.ndarray` with same shape as `context` in which
                 1-value indicate skip-gram, 0-value indicate negative nodes, thus, there
                 exists only one 1-value in each row of `labels`.
        """
        target, context = [], []
        # generate skip-gram from paths
        for path in paths:
            for i, wi in enumerate(path):
                window_start = max(0, i - self._window_size)
                window_end = min(len(path), i + self._window_size + 1)
                for j in range(window_start, window_end):
                    if j != i:
                        wj = path[j]
                        target.append(wi)
                        context.append(wj)
        # negative sampling
        ns = np.random.choice(
            self._vocabulary_size,
            size=(len(target), self._negative_samples+1),
            p=self._sampling_table,
        )

        # shuffle context & labels to random the place of neighbour node in context
        labels = np.zeros(shape=ns.shape, dtype=np.int32)
        col_indices = np.random.randint(self._negative_samples+1, size=len(target))
        labels[np.arange(labels.shape[0]), col_indices] = 1
        ns[np.arange(labels.shape[0]), col_indices] = context
        return (np.array(target, dtype=np.int64), ns), labels

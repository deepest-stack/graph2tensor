#!/usr/bin/env python3
from .base_explainer import BaseExplainer
import tensorflow as tf


class IntegratedGradients(BaseExplainer):
    r"""
    A GNN model explainer based on `IntegratedGradients <https://arxiv.org/abs/1703.01365>`__.

    :param model: object of :class:`graph2tensor.model.models.MessagePassing`, the model
                  to be explained
    :param n_steps: how many values to interpolate between baseline and instance
    """
    def __init__(self, model, n_steps):
        self.model = model
        self.n_steps = n_steps

    def interpolate_inputs(self, inputs):
        out = []
        for path in inputs:
            path_interpolated = []
            for src, edge, dst, segment_ids in path:
                weights = tf.linspace(
                    tf.zeros_like(segment_ids, dtype=tf.float32),
                    tf.ones_like(segment_ids, dtype=tf.float32),
                    num=self.n_steps
                )
                path_interpolated.append(
                    (src, edge, dst, segment_ids, weights))
            out.append(tuple(path_interpolated))
        return tuple(out)

    def explain(self, inputs, target_class):
        """
        Predict inputs using model and explain results

        :param inputs: instances to predict and explain
        :param target_class: index of target class
        :return: explanations and probabilities of target class
        """
        interpolated_inputs = self.interpolate_inputs(inputs)
        preds = []
        with tf.GradientTape(persistent=True) as tape:
            for path in interpolated_inputs:
                for _, _, _, _, weights in path:
                    tape.watch(weights)
            for i in range(self.n_steps):
                input_i = []
                for path in interpolated_inputs:
                    path_i = []
                    for src, edge, dst, segment_ids, weights in path:
                        path_i.append(
                            (src, edge, dst,
                             segment_ids, weights[i])
                        )
                    input_i.append(tuple(path_i))
                x = self.model.explain_call(tuple(input_i))
                preds.append(x[:, target_class])
            logits = x
            preds = tf.concat(preds, axis=0)
        grads = []
        for path1, path2 in zip(inputs, interpolated_inputs):
            path_grads = []
            for (_, edge1, _, segment_ids), (_, _, _, _, weights) in zip(path1, path2):
                edge_grads = tf.abs(
                    tf.reduce_sum(
                        tape.gradient(preds, weights),
                        axis=0,
                        keepdims=False
                    )
                )
                path_grads.append((edge1, edge_grads, segment_ids))
            grads.append(tuple(path_grads))
        return tuple(grads), tf.nn.softmax(logits, axis=1)[:, target_class]


#!/usr/bin/env python3
from graph2tensor.model.explainer import IntegratedGradients
from unittest import TestCase, main
from test_utils import graph_setup, build_model_for_arxiv
from graph2tensor.client import NumpyGraph
import numpy as np
from graph2tensor.model.data import build_output_signature, EgoTensorGenerator
import tensorflow as tf

g = NumpyGraph()
model = None


class TestIntegratedGradients(TestCase):
    def test_1_setup(self):
        global model
        graph_setup(g)
        model = build_model_for_arxiv()

    def test_integrated_gradients(self):
        baseline = {'feat': np.zeros(128), 'year': np.zeros(32)}
        explainer = IntegratedGradients(model, baseline, 10)
        ids = np.random.randint(169343, size=1024)
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
                args=(ids, 256, False, ids),
                output_signature=output_signature
            )
            for x, y in ds:
                explanation, probs = explainer.explain(x, 0)
                print(explanation)
                print(probs)


if __name__ == "__main__":
    main()

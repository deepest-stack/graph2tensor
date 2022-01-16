#!/usr/bin/env python3
from graph2tensor.interface import *
from graph2tensor.model.explainer import IntegratedGradients
from unittest import TestCase, main
from test_utils import build_model_for_arxiv
import json
from test_utils import graph_setup
from graph2tensor.client import NumpyGraph
g, model = None, None


class TestInterface(TestCase):

    def _test_1_build_graph_from_pg(self):
        nodes_def = {
            "paper": {
                "attrs": [("feat", "float[128]"), ("year", "int")],
                "labeled": True,
                "node_label": "label"
            }
        }
        edges_def = {
            "cites": {
                "src_type": "paper",
                "dst_type": "paper",
                "attrs": [],
                "directed": True,
            },
        }
        graph_def = {
            "graph_name": "ogbn_arxiv",
            "db_host": "192.168.100.21",
            "db_port": 15432,
            "db_database": "gnn_dev",
            "db_user": "gpadmin",
            "db_password": None,
            "nodes_def": nodes_def,
            "edges_def": edges_def
        }
        global g
        g = graph_from_pg(**graph_def)

    def test_1_setup(self):
        global g
        g = NumpyGraph()
        graph_setup(g)

    def test_1_build_model(self):
        global model
        model = build_model_for_arxiv()

    def test_graph_with_dgl(self):
        dgl_graph = graph_to_dgl(g)
        g1 = graph_from_dgl(dgl_graph)
        assert g.schema == g1.schema

    def test_train(self):
        dataset_def = {
            "valid_size": .25,
            "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
            "batch_size": 256,
            "include_edge": False,
            "shuffle": True,
            "sampler_process_num": 1,
            "converter_process_num": 2,
            "expand_factors": 10,
            "strategies": "random",
        }
        train_loop_def = {
            "optimizer": {"optimizer_class": "Adam"},
            "loss": {"loss_class": "SparseCategoricalCrossentropy", "from_logits": True},
            "metrics": [{"metric_class": "SparseCategoricalAccuracy"}, ],
            "epochs": 1,
            "checkpoint_save_freq": 1,
            "tensorboard_update_freq": 1
        }
        model_save_path = "/home/liusx/models/gcn_model_2"
        tensorboard_log_dir = "/home/liusx/tensorboard_logs/gcn_model_2"
        checkpiont_dir = "/home/liusx/ckpts/gcn_model_2"
        model_id, history = train(model, "gcn_model_2", g,
                                  np.arange(169343),
                                  np.random.randint(2, size=169343, dtype=np.int32),
                                  dataset_def, train_loop_def,
                                  checkpiont_dir, tensorboard_log_dir, model_save_path)
        print(history)

    def test_z_evaluate(self):
        dataset_def = {
            # "valid_size": .25,
            "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
            "batch_size": 256,
            "include_edge": False,
            # "shuffle": True,
            "sampler_process_num": 1,
            "converter_process_num": 2,
            "expand_factors": 10,
            "strategies": "random",
        }
        metrics = evaluate(model, g,
                           np.arange(169343),
                           np.random.randint(2, size=169343, dtype=np.int32),
                           dataset_def)

    def test_z_predict(self):
        dataset_def = {
            # "valid_size": .25,
            "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
            "batch_size": 256,
            "include_edge": False,
            # "shuffle": True,
            "sampler_process_num": 1,
            "converter_process_num": 2,
            "expand_factors": 10,
            "strategies": "random",
        }
        predictions = predict(model, g, np.arange(169343), dataset_def)

    def test_z_predict_and_explain(self):
        dataset_def = {
            # "valid_size": .25,
            "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
            "batch_size": 256,
            "include_edge": False,
            # "shuffle": True,
            "sampler_process_num": 1,
            "converter_process_num": 2,
            "expand_factors": 2,
            "strategies": "random",
        }
        explainer = IntegratedGradients(
            model,
            n_steps=32
        )
        for explanation in predict_and_explain(model, g,
                                               np.random.randint(169343, size=1024, dtype=np.int64),
                                               dataset_def, explainer, 24, 0.0):
            print(explanation[0])
            print(explanation[1])
            print(json.dumps(explanation[2], indent=4))


if __name__ == "__main__":
    main()

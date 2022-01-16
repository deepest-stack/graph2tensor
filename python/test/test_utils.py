#!/usr/bin/env python3
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from graph2tensor import build_model


def gen_node_attrs():
    node_attrs = {
        "nodeA": {}
    }
    node_attrs["nodeA"]["age"] = np.random.randint(18, 60, size=(1024, 1), dtype=np.int32)
    node_attrs["nodeA"]["emb"] = np.random.rand(1024, 64)
    return node_attrs


def gen_edge_attrs():
    edge_attrs = {
        "edgeA": {}
    }
    edge_attrs["edgeA"]["edge_type"] = np.random.randint(4, size=1024)
    return edge_attrs


def get_ogbn_arxiv():
    dataset = DglNodePropPredDataset(name="ogbn-arxiv", root='/home/liusx/dataset/')
    graph, label = dataset[0]
    uv = graph.edges('uv')
    src_ids, dst_ids = uv[0].numpy(), uv[1].numpy()
    paper_attrs_data = {
        "year": graph.ndata['year'].numpy(),
        "feat": graph.ndata['feat'].numpy(),
        "label": label.numpy()
    }
    return src_ids, dst_ids, paper_attrs_data


def graph_setup(graph):
    src_ids, dst_ids, paper_attrs_data = get_ogbn_arxiv()
    labels = paper_attrs_data.pop("label")
    graph.add_node("paper",
               [("year", "int"), ("feat", "float[128]")],
               labeled=True,
               node_label=labels,
               node_attrs=paper_attrs_data)
    graph.add_edge("cites", "paper", "paper", [],
                   src_ids=src_ids+1, dst_ids=dst_ids+1,
                   edge_probs=np.random.rand(src_ids.shape[0])
                   )


def build_model_for_arxiv():
    model_def = {
        "conv_layers_def": [
            [{"layer_class": "GCNConv", "units":128, "add_self_loop":True, "activation":'relu', "name": "gcn1"},
             {"layer_class": "GCNConv", "units":64, "add_self_loop":True, "activation":'relu', "name": "gcn2"}],
        ],
        "pre_proc_layers_def": [
            {"layer_class": "IntegerLookupEncoder",
             "name": "year_lookup",
             "cate_feat_def": [{"attr_name": "year", "vocabulary": [1971, 1986, 1987, 1988]+list(range(1990, 2021))}, ]
            },
            {"layer_class": 'EmbeddingEncoder',
             "name": "year_emb",
             "cate_feat_def": [{"attr_name": "year", "input_dim": 37, "output_dim": 32}, ]
            }
        ],
        "reduce_layer_def": {"layer_class": "SumPathReduce"},
        "out_layers_def": [{"layer_class": "Dense", "name": "out1", "units": 40}, ],
        "concat_hidden": False,
        "attr_reduce_mode": 'concat',
        "name": "gcn_model"
    }
    return build_model(**model_def)


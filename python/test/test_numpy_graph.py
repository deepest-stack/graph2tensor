#!/usr/bin/env python3

from graph2tensor.client import NumpyGraph
from graph2tensor.common import Nodes, Edges
from unittest import TestCase, main
import numpy as np
from test_utils import gen_node_attrs, gen_edge_attrs


g = NumpyGraph()


class TestNumpyGraph(TestCase):

    def test_1_add_node(self):
        node_attrs = gen_node_attrs()
        g.add_node("nodeA",
                   attrs_info=[("age", "int"), ("emb", "float[64]")],
                   node_attrs=node_attrs["nodeA"],
                   labeled=True,
                   node_label=np.random.randint(10, size=1024)
                   )
        g.add_node("nodeB", [], labeled=False)

    def test_1_add_edge(self):
        edge_attrs = gen_edge_attrs()
        ids = np.zeros((1, 1), dtype=np.int64)
        while ids.shape[1] < 1024:
            ids = np.random.randint(1, 1023, size=(2, 2048))
            src_ids, dst_ids = np.min(ids, axis=0)-1, np.max(ids, axis=0)+1
            ids = np.unique(np.concatenate((src_ids, dst_ids)).reshape((2, 2048)), axis=1)[:, :1024]
        src_ids, dst_ids = ids[0], ids[1]
        edge_ids = np.arange(1025)
        with self.assertRaises(ValueError):
            g.add_edge("edgeA", "nodeA", "nodeA", [("edge_type", "int")],
                       directed=False,
                       src_ids=src_ids,
                       dst_ids=dst_ids,
                       edge_ids=edge_ids)
        edge_ids = np.arange(1024)
        with self.assertRaises(ValueError):
            g.add_edge("edgeA", "nodeA", "nodeB", [("edge_type", "int")],
                       directed=False,
                       src_ids=src_ids,
                       dst_ids=dst_ids,
                       edge_ids=edge_ids)
        g.add_edge("edgeA", "nodeA", "nodeA", [("edge_type", "int")],
                   directed=False,
                   src_ids=src_ids,
                   dst_ids=dst_ids,
                   edge_ids=edge_ids,
                   edge_attrs=edge_attrs["edgeA"],
                   edge_probs=np.random.rand(1024)
                   )
        g.add_edge("edgeB", "nodeA", "nodeB", [], src_ids=src_ids, dst_ids=dst_ids)

    def test_serialize(self):
        _ = NumpyGraph.from_config(g.to_config())

    def test_get_src_type(self):
        assert g.get_src_type("edgeA") == "nodeA"
        assert g.get_src_type("edgeB") == "nodeA"

    def test_get_dst_type(self):
        assert g.get_dst_type("edgeA") == "nodeA"
        assert g.get_dst_type("edgeB") == "nodeB"

    def test_get_edge_attrs(self):
        assert g.get_edge_attr_info("edgeA") == {"edge_type": "int"}
        assert g.get_edge_attr_info("edgeB") == {}

    def test_is_edge_directed(self):
        assert g.is_edge_directed("edgeA") is False
        assert g.is_edge_directed("edgeB")

    def test_get_node_attrs(self):
        assert g.get_node_attr_info("nodeA") == dict([("age", "int"), ("emb", "float[64]")])
        assert g.get_node_attr_info("nodeB") == {}

    def test_is_node_labeled(self):
        assert g.is_node_labeled("nodeA")
        assert not g.is_node_labeled("nodeB")

    def test_get_node_label(self):
        g.get_node_label("nodeA")
        with self.assertRaises(ValueError):
            g.get_node_label("nodeB")

    def test_lookup_nodes(self):
        _ = g.lookup_nodes(Nodes("nodeA", np.random.randint(1024, size=32)))
        _ = g.lookup_nodes(Nodes("nodeB", np.random.randint(1024, size=32)))

    def test_lookup_edges(self):
        _ = g.lookup_edges(Edges("edgeA", edge_ids=np.random.randint(1024, size=32)))
        _ = g.lookup_edges(Edges("edgeB", edge_ids=np.random.randint(1024, size=32)))

    def test_sample_neighbors(self):
        nodeA_max_id = g._relation_store._edges["edgeA"]["adj_mat"].shape[0]
        _ = g.sample_neighbors(
            np.random.randint(nodeA_max_id, size=32),
            "edgeA",
            strategy='all'
        )
        _ = g.sample_neighbors(
            np.random.randint(nodeA_max_id, size=32),
            "edgeA",
            num=4,
            strategy='topk'
        )
        _ = g.sample_neighbors(
            np.random.randint(nodeA_max_id, size=32),
            "edgeA",
            num=4,
            strategy='random'
        )
        with self.assertRaises(Warning):
            _ = g.sample_neighbors(
                np.random.randint(nodeA_max_id, size=32),
                "edgeB",
                num=4,
                strategy='random',
                use_edge_probs=True
            )
        with self.assertRaises(ValueError):
            _ = g.sample_neighbors(
                np.random.randint(nodeA_max_id, size=32),
                "edgeB",
                num=4,
                strategy='topk'
            )

    def test_random_walk(self):
        nodeA_max_id = g._relation_store._edges["edgeA"]["adj_mat"].shape[0]
        _ = g.random_walk(
            np.arange(nodeA_max_id),
            "edgeA",
            walk_length=10
        )
        with self.assertRaises(Warning):
            _ = g.random_walk(
                np.arange(nodeA_max_id),
                "edgeB",
                walk_length=10,
                use_edge_probs=True
            )
        _ = g.random_walk(
            np.arange(nodeA_max_id),
            "edgeA",
            walk_length=10,
            discard_frequent_nodes=True,
            freq_th=2/1024
        )

    def test_node2vec_walk(self):
        nodeA_max_id = g._relation_store._edges["edgeA"]["adj_mat"].shape[0]
        _ = g.node2vec_walk(
            np.arange(nodeA_max_id),
            "edgeA",
            walk_length=10,
            p=.5,
            q=2.
        )
        with self.assertRaises(Warning):
            _ = g.node2vec_walk(
                np.arange(nodeA_max_id),
                "edgeB",
                walk_length=10,
                p=.5,
                q=2.,
                use_edge_probs=True
            )
        _ = g.node2vec_walk(
            np.arange(nodeA_max_id),
            "edgeA",
            walk_length=10,
            p=.5,
            q=2.,
            discard_frequent_nodes=True,
            freq_th=2/1024
        )


if __name__ == "__main__":
    main()

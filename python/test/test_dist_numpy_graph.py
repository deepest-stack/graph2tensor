#!/usr/bin/env python3
from test_utils import get_ogbn_arxiv
from graph2tensor.client.distributed import *
from graph2tensor.common import Nodes
from unittest import TestCase, main
import numpy as np
from concurrent import futures
import grpc
s1 = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
s2 = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
g = DistNumpyGraph()


class TestDistNumpyGraph(TestCase):

    def test_1_start_servicer(self):
        src_ids, dst_ids, paper_data = get_ogbn_arxiv()
        cites_servicer = DistNumpyRelationStoreServicer(src_ids+1, dst_ids+1)
        paper_servicer = DistNumpyAttributeStoreServicer()
        paper_servicer.add_attr('year', paper_data['year'])
        paper_servicer.add_attr('feat', paper_data['feat'])
        paper_servicer.add_label(paper_data['label'])
        add_relation_servicer(cites_servicer, s1)
        add_attribute_servicer(paper_servicer, s2)
        s1.add_insecure_port('[::]:12345')
        s2.add_insecure_port('[::]:12346')
        s1.start()
        s2.start()

    def test_add_node(self):
        g.add_node("paper",[("year","int"),("feat","float[128]")],True, target="127.0.0.1:12346")

    def test_add_edge(self):
        g.add_edge("cites","paper","paper", [], relation_target="127.0.0.1:12345")

    def test_lookup(self):
        _ = g.lookup_nodes(Nodes("paper", np.random.randint(169242, size=1024)))

    def test_get_node_attr_data(self):
        for x in g.get_node_attr_data("paper", "feat"):
            assert x.shape[1] == 128
        for x in g.get_node_attr_data("paper", "year"):
            assert x.shape[1] == 1
        with self.assertRaises(ValueError):
            for _ in g.get_node_attr_data("nonexistent_node", "feat"):
                pass
        with self.assertRaises(ValueError):
            for _ in g.get_node_attr_data("paper", "nonexistent_attr"):
                pass

    def test_get_node_label(self):
        for _ in g.get_node_label("paper"):
            pass
        with self.assertRaises(ValueError):
            _ = g.get_node_label("nonexistent_node")

    def test_get_edge_ids(self):
        for src_ids, dst_ids, edge_ids in g.get_edge_ids('cites'):
            assert src_ids.shape[0] == dst_ids.shape[0] == edge_ids.shape[0]
        with self.assertRaises(ValueError):
            for _ in g.get_edge_ids('nonexistent_edge'):
                pass

    def test_sample_neighbors(self):
        ids = np.random.randint(169242, size=1024)
        _ = g.sample_neighbors(
            ids, 'cites', strategy="all"
        )
        _ = g.sample_neighbors(
            ids, 'cites', 10, strategy="random"
        )
        _ = g.sample_neighbors(
            ids, 'cites', 10, strategy="random",
            use_edge_probs=True
        )

    def test_config(self):
        g1 = DistNumpyGraph.from_config(g.to_config())
        assert g1.schema == g.schema
        assert g1._stub_targets == g._stub_targets

    def test_z_stop_servicer(self):
        s1.stop(grace=None)
        s2.stop(grace=None)

    def test_random_walk(self):
        ids = np.random.randint(169242, size=1024)
        _ = g.random_walk(ids, 'cites', walk_length=10)

    def test_node2vec_walk(self):
        ids = np.random.randint(169242, size=1024)
        _ = g.node2vec_walk(ids, 'cites', walk_length=10, p=.5, q=2.)


if __name__ == "__main__":
    main()

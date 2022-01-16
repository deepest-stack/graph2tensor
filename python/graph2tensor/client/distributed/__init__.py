#!/usr/bin/env python3
import sys
import os.path as osp

print(osp.dirname(__file__))
sys.path.insert(0, osp.dirname(__file__))

from .dist_numpy_relation_store import DistNumpyRelationStoreServicer
from .dist_numpy_attribute_store import DistNumpyAttributeStoreServicer
from .dist_numpy_graph import DistNumpyGraph
from .dist_numpy_relation_store_pb2_grpc \
    import add_DistNumpyRelationStoreServicer_to_server as add_relation_servicer
from .dist_numpy_attribute_store_pb2_grpc \
    import add_DistNumpyAttributeStoreServicer_to_server as add_attribute_servicer

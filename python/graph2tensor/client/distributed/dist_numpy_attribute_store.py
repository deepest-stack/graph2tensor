#!/usr/bin/env python3
import dist_numpy_attribute_store_pb2_grpc
import numpy as np
import math
from dist_numpy_attribute_store_pb2 import AttrProto
from ndarray_pb2 import NdarrayProto


class DistNumpyAttributeStoreServicer(dist_numpy_attribute_store_pb2_grpc.DistNumpyAttributeStoreServicer):
    """
    The server for node & edge attributes fetching.
    """
    def __init__(self):
        super(DistNumpyAttributeStoreServicer, self).__init__()
        self._attrs = {}

    def add_attr(self, attr_name, attr_data):
        """
        Add a new attribute into node or edge.

        :param attr_name: attribute name
        :param attr_data: attribute data in `np.narray` format
        """
        self._attrs[attr_name] = attr_data

    def add_label(self, labels):
        """
        Add label into node.

        :param labels: labels of node in `np.narray` format
        """
        self.add_attr("label", labels)

    def lookup(self, request, context):
        ids = np.frombuffer(request.array_content, dtype=request.dtype)
        attrs = {}
        for k in self._attrs:
            x = np.take(self._attrs[k], ids, axis=0)
            v = NdarrayProto(dtype=x.dtype.__str__(), array_content=x.tobytes())
            v.dims.extend(x.shape)
            attrs[k] = v
        return AttrProto(attrs=attrs)

    def get_attr_data(self, request, context):
        total_n = self._attrs[request.name].shape[0]
        for i in range(math.ceil(total_n/request.batch_size)):
            array_content = self._attrs[request.name][i*request.batch_size:(i+1)*request.batch_size]
            x = NdarrayProto(
                dtype=array_content.dtype.__str__(),
                array_content=array_content.tobytes())
            x.dims.extend(array_content.shape)
            yield x

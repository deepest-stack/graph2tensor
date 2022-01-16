#!/usr/bin/env python3
import regex
import tensorflow as tf
from inspect import isgenerator
from collections import Counter
import numpy as np
from ...converter.ego2tensor import NID, EID, UID, VID


def build_output_signature(schema, meta_paths, include_edge, add_label=True):

    map2tf = {"str": tf.string, "float": tf.float32, "int": tf.int32}

    def _parse_attr_type(attr_type):
        if attr_type in ("str", "float", "int"):
            return attr_type, 1
        if attr_type.startswith("str"):
            return "str", int(attr_type[4:-1])
        if attr_type.startswith("float"):
            return "float", int(attr_type[6:-1])
        if attr_type.startswith("int"):
            return "int", int(attr_type[4:-1])
        raise ValueError("Unrecognized attr_type: `%s`" % attr_type)

    def _build_node(node_type):
        sign = dict()
        component_attr_type = schema["nodes"][node_type]['attrs']
        for attr_name, attr_type in component_attr_type.items():
            dtype, dim = _parse_attr_type(attr_type)
            sign[attr_name] = tf.TensorSpec(shape=(None, dim), dtype=map2tf[dtype])
        sign[NID] = tf.TensorSpec(shape=(None, ), dtype=tf.int64)
        return sign

    def _build_edge(edge_type):
        edge_sign = {
            EID: tf.TensorSpec(shape=(None, ), dtype=tf.int64),
            UID: tf.TensorSpec(shape=(None, ), dtype=tf.int64),
            VID: tf.TensorSpec(shape=(None, ), dtype=tf.int64),
        }
        if include_edge:
            component_attr_type = schema["edges"][edge_type]['attrs']
            for attr_name, attr_type in component_attr_type.items():
                dtype, dim = _parse_attr_type(attr_type)
                edge_sign[attr_name] = tf.TensorSpec(shape=(None, dim), dtype=map2tf[dtype])
        return edge_sign

    def _build(hop):
        src, edge, dst = hop
        src_sign = _build_node(src)
        dst_sign = _build_node(dst)
        edge_sign = _build_edge(edge)
        segment_sign = tf.TensorSpec(shape=(None,), dtype=tf.int64)
        return src_sign, edge_sign, dst_sign, segment_sign

    pt = regex.compile("\((\S+?)\)\s*-\[(\S+?)\]-\s*\((\S+?)\)")
    paths_parsed = [pt.findall(path, overlapped=True) for path in meta_paths]
    signature = tuple([tuple([_build(hop) for hop in hops]) for hops in paths_parsed])
    # add label
    if add_label:
        signature = (signature, tf.TensorSpec(shape=(None,), dtype=tf.int32))
    return signature


def build_sampling_table(graph, edge_type, num_nodes=None, power=.75):
    edges = graph.get_edge_ids(edge_type)
    freq = Counter()
    if isgenerator(edges):
        for src_ids, dst_ids, _ in edges:
            freq.update(src_ids)
            freq.update(dst_ids)
    else:
        src_ids, dst_ids, _ = edges
        freq.update(src_ids)
        freq.update(dst_ids)
    if not num_nodes:
        num_nodes = max(freq.keys()) + 1
    sampling_table = np.zeros(num_nodes)
    sampling_table[list(freq.keys())] = list(freq.values())
    sampling_table = np.power(sampling_table, power)
    return sampling_table / sampling_table.sum()


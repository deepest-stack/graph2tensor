#!/usr/bin/env python3
import os
from typing import Dict, Optional, Any, Tuple, Sequence
from .client import NumpyGraph
from .model.models import MessagePassing
from .model import layers
from .model.data import EgoTensorGenerator, build_output_signature
from tensorflow.keras import layers as keras_layers
import psycopg2
import pandas as pd
import numpy as np
import tensorflow as tf
import os.path as osp
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import dgl
import re
from .converter.ego2tensor import EID, UID, VID
from copy import deepcopy


def graph_to_dgl(graph: NumpyGraph) -> dgl.DGLGraph:
    """
    Transfer a :class:`graph2tensor.client.NumpyGraph` object into a
    heterogeneous & directed `DGLGraph`.
    All the nodes & edges attributes will be put into nodes & edges data, the
    node label (if had) will also be put into nodes data with key "label", and
    the edge probabilities (if had) into edges data with key "edge_probs".

    :param graph: a :class:`graph2tensor.client.NumpyGraph` object
    :return: a `DGLGraph` object
    """
    graph_data = {}
    edge_ids = {}
    for edge_type, edge_info in graph.schema["edges"].items():
        src_ids, dst_id, eids = graph.get_edge_ids(edge_type)
        graph_data[(edge_info["src_type"], edge_type, edge_info["dst_type"])] = \
            (dgl.backend.tensor(src_ids), dgl.backend.tensor(dst_id))
        edge_ids[edge_type] = eids
    dgl_graph = dgl.heterograph(graph_data)
    for edge_type, edge_info in graph.schema["edges"].items():
        for attr_name in edge_info["attrs"]:
            dgl_graph.edges[edge_type].data[attr_name] = dgl.backend.tensor(
                graph.get_edge_attr_data(edge_type, attr_name)[edge_ids[edge_type]]
            )
        edge_probs = graph.get_edge_probs(edge_type)
        if edge_probs is not None:
            dgl_graph.edges[edge_type].data["edge_probs"] = dgl.backend.tensor(
                edge_probs
            )
    for node_type, node_info in graph.schema["nodes"].items():
        for attr_name in node_info["attrs"]:
            dgl_graph.nodes[node_type].data[attr_name] = dgl.backend.tensor(
                graph.get_node_attr_data(node_type, attr_name)
            )
        if node_info["labeled"]:
            dgl_graph.nodes[node_type].data["label"] = dgl.backend.tensor(
                graph.get_node_label(node_type)
            )
    return dgl_graph


def graph_from_dgl(dgl_graph: dgl.DGLGraph) -> NumpyGraph:
    """
    Transfer a `DGLGraph` object into :class:`graph2tensor.client.NumpyGraph`.
    All the nodes & edges data will be put into nodes & edges attributes,
    except for node data "label" and edge data "edge_probs", which will
    be taken as node label and edge probabilities respectively.

    :param dgl_graph: a `DGLGraph` object
    :return: a :class:`graph2tensor.client.NumpyGraph` object
    """
    def _build_attr_type(attr_schema):
        shape = attr_schema.shape[0]
        if "float" in attr_schema.dtype.__str__():
            dtype = "float"
        else:
            dtype = "int"
        if shape > 1:
            dtype = dtype + f"[{shape}]"
        return dtype

    g = NumpyGraph()
    dgl_graph = dgl_graph.to('cpu')
    for node_type in dgl_graph.ntypes:
        attrs_info = [(k, _build_attr_type(v))
                      for k, v in dgl_graph.node_attr_schemes(node_type).items()
                      if k != "label"]
        node_attrs = {k: v.numpy() for k, v in dgl_graph.nodes[node_type].data.items()}
        labeled, node_label = False, None
        if "label" in node_attrs:
            labeled = True
            node_label = node_attrs.pop("label")
        g.add_node(node_type, attrs_info=attrs_info, node_attrs=node_attrs,
                   labeled=labeled, node_label=node_label)
    for src_type, edge_type, dst_type in dgl_graph.canonical_etypes:
        src_ids, dst_ids = dgl_graph.edges(etype=edge_type, form='uv')
        attrs_info = [(k, _build_attr_type(v))
                      for k, v in dgl_graph.edge_attr_schemes(edge_type).items()
                      if k != "edge_probs"]
        kwargs = {}
        edge_attrs = {k: v.numpy() for k, v in dgl_graph.edges[edge_type].data.items()}
        if edge_attrs:
            kwargs["edge_attrs"] = edge_attrs
        if "edge_probs" in dgl_graph.edge_attr_schemes(edge_type):
            kwargs["edge_probs"] = dgl_graph.edges[edge_type].data["edge_probs"].numpy()
        g.add_edge(edge_type, src_type, dst_type,
                   attrs_info=attrs_info,
                   directed=True,
                   src_ids=src_ids.numpy(),
                   dst_ids=dst_ids.numpy(),
                   **kwargs
                   )
    return g


def graph_from_pg(graph_name: str,
                  nodes_def: Dict[str, Any],
                  edges_def: Dict[str, Any],
                  db_host: str,
                  db_port: int,
                  db_database: str,
                  db_user: str,
                  db_password: Optional[str],
                  **kwargs) -> NumpyGraph:
    """
    Build :class:`graph2tensor.client.NumpyGraph` from Postgres or Greenplum.

    :param graph_name: database schema, it suppose that each graph has it own schema.
    :param nodes_def: nodes definitions, see example below for more details
    :param edges_def: edges definitions, see example below for more details
    :param db_host: host of database
    :param db_port: port of database
    :param db_database: name of database instance
    :param db_user: user of database
    :param db_password: password of database
    :return: a :class:`graph2tensor.client.NumpyGraph` object

    :Example:
    >>> nodes_def = {
    ...     "nodeA": {
    ...         "attrs": [("attr1", "int"), ("attr2", "float"), ("attr3", "str")],
    ...         "labeled": True,
    ...         "node_label": "label"
    ...     },
    ...     "nodeB": {
    ...         "attrs": [("attr1", "int"), ("attr2", "float"), ("attr3", "str")],
    ...         "labeled": False
    ...     }
    ... }
    >>> edges_def = {
    ...     "edgeA": {
    ...         "src_type": "nodeA",
    ...         "dst_type": "nodeA",
    ...         "directed": True,
    ...         "attrs": [("attr1", "int"), ("attr2", "float"), ("attr3", "str")],
    ...         "edge_probs": "prob_column"
    ...     },
    ...     "edgeB": {
    ...         "src_type": "nodeA",
    ...         "dst_type": "nodeB",
    ...         "directed": False,
    ...         "attrs": [],
    ...     }
    ... }
    >>> graph_from_pg(
    ...     graph_name="graph1",
    ...     nodes_def=nodes_def,
    ...     edges_def=edges_def,
    ...     db_host="127.0.0.1",
    ...     db_port=15432,
    ...     db_database="gnn_dev",
    ...     db_user="gpadmin",
    ...     db_password=None,
    ...     default_int_attr=0,
    ...     default_float_attr=0.0,
    ...     default_str_attr=''
    ... )
    """
    map2np = {"str": np.str, "float": np.float32, "int": np.int32}
    g = NumpyGraph()
    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_database, user=db_user, password=db_password)
    with conn.cursor() as cur:
        # nodes_def = {
        #     "nodeA": {
        #         "attrs": [("attr1", "int"), ("attr2", "float"), ("attr3", "str")],
        #         "labeled": True,
        #         "node_label": "label"
        #     },
        #     "nodeB": {
        #         "attrs": [("attr1", "int"), ("attr2", "float"), ("attr3", "str")],
        #         "labeled": False
        #     }
        # }
        for node_type in nodes_def:
            attr_columns = [x[0] for x in nodes_def[node_type]["attrs"]]
            node_attrs = None
            if attr_columns:
                sql = f"SELECT %s FROM {graph_name}.vertex_{node_type} ORDER BY vid"
                cur.execute(sql % (", ".join(attr_columns),))
                df = pd.DataFrame(cur.fetchall(), columns=attr_columns)
                node_attrs = {col: np.array(df[col].tolist(), dtype=map2np[t.split('[')[0].strip()])
                              for col, t in nodes_def[node_type]["attrs"]}
            node_label = None
            if nodes_def[node_type]["labeled"]:
                sql = f"SELECT %s FROM {graph_name}.vertex_{node_type} ORDER BY vid"
                cur.execute(sql % (nodes_def[node_type]["node_label"],))
                node_label = np.array(cur.fetchall(), dtype=np.int32)
            g.add_node(node_type, attrs_info=nodes_def[node_type]["attrs"], node_attrs=node_attrs,
                       labeled=nodes_def[node_type]["labeled"], node_label=node_label)
        # edges_def = {
        #     "edgeA": {
        #         "src_type": "nodeA",
        #         "dst_type": "nodeA",
        #         "directed": True,
        #         "attrs": [("attr1", "int"), ("attr2", "float"), ("attr3", "str")],
        #         "edge_probs": "prob_column"
        #     },
        #     "edgeB": {
        #         "src_type": "nodeA",
        #         "dst_type": "nodeB",
        #         "directed": False,
        #         "attrs": [],
        #     }
        # }
        sql = f"SELECT %s FROM {graph_name}.edge_%s"
        for edge_type in edges_def:
            columns = ["src_id", "dst_id"]
            columns.extend([x[0] for x in edges_def[edge_type]["attrs"]])
            if edges_def[edge_type].get("edge_probs"):
                columns.append(edges_def[edge_type]["edge_probs"])
            cur.execute(sql % (", ".join(columns), edge_type))
            df = pd.DataFrame(cur.fetchall(), columns=columns)
            kwargs = {}
            if edges_def[edge_type].get("edge_probs"):
                kwargs["edge_probs"] = df[edges_def[edge_type]["edge_probs"]].to_numpy(dtype=np.float64)
            if edges_def[edge_type]["attrs"]:
                kwargs["edge_attrs"] = {col: np.array(df[col].tolist(), dtype=map2np[t.split('[')[0].strip()])
                                        for col, t in edges_def[edge_type]["attrs"]}
            g.add_edge(edge_type, edges_def[edge_type]["src_type"], edges_def[edge_type]["dst_type"],
                       attrs_info=edges_def[edge_type]["attrs"],
                       directed=edges_def[edge_type]["directed"],
                       src_ids=df["src_id"].to_numpy(dtype=np.int64),
                       dst_ids=df["dst_id"].to_numpy(dtype=np.int64),
                       **kwargs
                       )
    conn.close()
    return g


def build_model(name: Optional[str],
                conv_layers_def: Sequence[Sequence[Dict[str, Any]]],
                attr_reduce_mode: Optional[str],
                pre_proc_layers_def: Optional[Sequence[Dict[str, Any]]],
                reduce_layer_def: Optional[Dict[str, Any]],
                out_layers_def: Optional[Sequence[Dict[str, Any]]],
                concat_hidden: bool) -> MessagePassing:
    """
    Build a :class:`graph2tensor.model.models.MessagePassing` model.

    :param name: name of model
    :param conv_layers_def: convolution layers definitions, see example below for more details
    :param attr_reduce_mode: how to reduce attributes of nodes/edges, expected 'concat',
                             'mean' or 'sum'
    :param pre_proc_layers_def: pre-processing layers definitions, `None` means no pre-processing
                                layer, see example below for more details
    :param reduce_layer_def: reduce layer definition, see example below and
                             :class:`graph2tensor.model.models.MessagePassing` for more details
    :param out_layers_def: out layers definitions, `None` means no out layers,
                            see example below for more details
    :param concat_hidden: whether or not concatenate outputs of convolution layers
    :return: a :class:`graph2tensor.model.models.MessagePassing` object

    :Example:
    >>> conv_layers_def = [
    ...     [{"layer_class": "GCNConv",
    ...       "input_dims": 128+32,
    ...       "units": 128,
    ...       "add_self_loop": True,
    ...       "activation": "relu",
    ...       "name": "gcn1"},
    ...      {"layer_class": "GCNConv",
    ...       "input_dims":128,
    ...       "units":64,
    ...       "add_self_loop":True,
    ...       "activation":'relu',
    ...       "name": "gcn2"}],
    ... ]
    >>> pre_proc_layers_def = [
    ...     {"layer_class": "IntegerLookupEncoder",
    ...      "name": "year_lookup",
    ...      "cate_feat_def": [{"attr_name": "year", "vocabulary": [1971, 1986, 1987, 1988]+list(range(1990, 2021))}, ]
    ...      },
    ...     {"layer_class": 'EmbeddingEncoder',
    ...      "name": "year_emb",
    ...      "cate_feat_def": [{"attr_name": "year", "input_dim": 37, "output_dim": 32}, ]
    ...      }
    ... ]
    >>> reduce_layer_def = {"layer_class": "Maximum"}
    >>> out_layers_def = [{"layer_class": "Dense", "name": "out1", "units": 40}, ]
    >>> build_model(
    ...     "gcn_model",
    ...     conv_layers_def,
    ...     'concat',
    ...     pre_proc_layers_def,
    ...     reduce_layer_def,
    ...     out_layers_def,
    ...     concat_hidden=False
    ... )
    """

    def _getattr(cls):
        try:
            return getattr(layers, cls)
        except AttributeError:
            return getattr(keras_layers, cls)

    # layers_def = [["GCNConv", {"input_dims": 16, "units": 32}],
    #               ["GCNConv", {"input_dims": 32, "units": 16}]]
    conv_layers = [[_getattr(layer_def.pop("layer_class"))(**layer_def) for layer_def in branch]
                   for branch in deepcopy(conv_layers_def)]

    if pre_proc_layers_def:
        pre_proc_layers = [_getattr(layer_def.pop("layer_class"))(**layer_def)
                           for layer_def in deepcopy(pre_proc_layers_def)]
    else:
        pre_proc_layers = None

    if reduce_layer_def:
        reduce_layer_def_ = deepcopy(reduce_layer_def)
        reduce_layer = _getattr(
            reduce_layer_def_.pop("layer_class")
        )(**reduce_layer_def_)
    else:
        reduce_layer = None

    if out_layers_def:
        out_layers = [_getattr(layer_def.pop("layer_class"))(**layer_def)
                      for layer_def in deepcopy(out_layers_def)]
    else:
        out_layers = None

    return MessagePassing(
        conv_layers=conv_layers,
        attr_reduce_mode=attr_reduce_mode,
        pre_proc_layers=pre_proc_layers,
        reduce_layer=reduce_layer,
        out_layers=out_layers,
        name=name,
        concat_hidden=concat_hidden
    )


def train(model, model_id, graph, seed_ids, seed_labels,
          dataset_def, train_loop_def, checkpoint_dir,
          tensorboard_log_dir, model_save_path=None):
    """
    Train a model.

    :param model: :class:`graph2tensor.model.models.MessagePassing` object, or other
                  equivalent object of subclass of `tf.keras.Model`
    :param model_id: id of model
    :param graph: a :class:`graph2tensor.client.NumpyGraph` object to generate
                  dataset from
    :param seed_ids: ids of seed nodes in `np.1darray`
    :param seed_labels: labels of seed nodes in `np.1darray`
    :param dataset_def: dataset definition, see example below and
                        :class:`graph2tensor.model.data.DataGenerator` for more details
    :param train_loop_def: train loop definition, see example below for more details
    :param checkpoint_dir: directory to save checkpoints files
    :param tensorboard_log_dir: directory to save tensorboard summary files
    :param model_save_path: model save path
    :return: model_id passed and training history

    :Example:
    >>> dataset_def = {
    ...     "valid_size": .25,
    ...     "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
    ...     "batch_size": 256,
    ...     "include_edge": False,
    ...     "shuffle": True,
    ...     "sampler_process_num": 1,
    ...     "converter_process_num": 2,
    ...     "expand_factors": 10,
    ...     "strategies": "random",
    ... }
    >>> train_loop_def = {
    ...     "optimizer": {"optimizer_class": "Adam"},
    ...     "loss": {"loss_class": "SparseCategoricalCrossentropy", "from_logits": True},
    ...     "metrics": [{"metric_class": "SparseCategoricalAccuracy"}, ],
    ...     "epochs": 1,
    ...     "checkpoint_save_freq": 1,
    ...     "tensorboard_update_freq": 1,
    ...     "class_weight": {0: 1, 1: 6}
    ... }
    >>> model_id, history = train(model, "model_12345678", graph, seed_ids, seed_labels,
    ...     dataset_def, train_loop_def, "models/model_12345678", "tensorboard_log/model_12345678")
    """

    train_seed_ids, valid_seed_ids, train_seed_labels, valid_seed_labels = train_test_split(
        seed_ids, seed_labels,
        test_size=dataset_def["valid_size"],
        shuffle=dataset_def["shuffle"]
    )
    output_signature = build_output_signature(
        graph.schema,
        dataset_def["meta_paths"],
        include_edge=dataset_def["include_edge"]
    )
    opt_def = deepcopy(train_loop_def["optimizer"])
    loss_def = deepcopy(train_loop_def["loss"])
    model.compile(
        optimizer=getattr(tf.keras.optimizers, opt_def.pop("optimizer_class"))(**opt_def),
        loss=getattr(tf.keras.losses, loss_def.pop("loss_class"))(**loss_def),
        metrics=[getattr(tf.keras.metrics, m_def.pop("metric_class"))(**m_def)
                 for m_def in deepcopy(train_loop_def["metrics"])]
    )
    if not osp.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    with EgoTensorGenerator(graph, **dataset_def) as data_gen:
        train_dataset = tf.data.Dataset.from_generator(
            data_gen,
            args=(train_seed_ids,
                  dataset_def["batch_size"],
                  False,
                  train_seed_labels
                  ),
            output_signature=output_signature
        )
        valid_dataset = tf.data.Dataset.from_generator(
            data_gen,
            args=(valid_seed_ids,
                  dataset_def["batch_size"],
                  False,
                  valid_seed_labels
                  ),
            output_signature=output_signature
        )
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=train_loop_def["epochs"],
            class_weight=train_loop_def.get("class_weight"),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=osp.join(checkpoint_dir, "{epoch:02d}"),
                    save_weights_only=True,
                    save_freq=train_loop_def["checkpoint_save_freq"]),
                tf.keras.callbacks.TensorBoard(
                    log_dir=tensorboard_log_dir,
                    write_graph=False,
                    update_freq=train_loop_def["tensorboard_update_freq"]
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=train_loop_def.get("patience", 0),
                    restore_best_weights=True
                )
            ]
        ).history

    if model_save_path:
        if not osp.exists(model_save_path):
            os.mkdir(model_save_path)
        model.save_weights(model_save_path)
    return model_id, history


def evaluate(model, graph, seed_ids, seed_labels, dataset_def):
    """
    Evaluate a model.

    :param model: :class:`graph2tensor.model.models.MessagePassing` object, or other
                  equivalent object of subclass of `tf.keras.Model`
    :param graph: a :class:`graph2tensor.client.NumpyGraph` object to generate
                  dataset from
    :param seed_ids: ids of seed nodes in `np.1darray`
    :param seed_labels: labels of seed nodes in `np.1darray`
    :param dataset_def: dataset definition, see example below and
                        :class:`graph2tensor.model.data.DataGenerator` for more details
    :return: the value of metrics

    :Example:
    >>> dataset_def = {
    ...     "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
    ...     "batch_size": 256,
    ...     "include_edge": False,
    ...     "sampler_process_num": 1,
    ...     "converter_process_num": 2,
    ...     "expand_factors": 10,
    ...     "strategies": "random",
    ... }
    >>> metrics = evaluate(model, graph, seed_ids, seed_labels, dataset_def)
    """
    output_signature = build_output_signature(
        graph.schema,
        dataset_def["meta_paths"],
        include_edge=dataset_def["include_edge"]
    )
    with EgoTensorGenerator(graph, **dataset_def) as data_gen:
        ds = tf.data.Dataset.from_generator(
            data_gen,
            args=(seed_ids,
                  dataset_def["batch_size"],
                  False,
                  seed_labels
                  ),
            output_signature=output_signature
        )
        return model.evaluate(ds)


def predict(model, graph, seed_ids, dataset_def):
    """
    Predict using a model.

    :param model: :class:`graph2tensor.model.models.MessagePassing` object, or other
                  equivalent object of subclass of `tf.keras.Model`
    :param graph: a :class:`graph2tensor.client.NumpyGraph` object to generate
                  dataset from
    :param seed_ids: ids of seed nodes in `np.1darray`
    :param dataset_def: dataset definition, see example below and
                        :class:`graph2tensor.model.data.DataGenerator` for more details
    :return: the prediction results

    :Example:
    >>> dataset_def = {
    ...     "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
    ...     "batch_size": 256,
    ...     "include_edge": False,
    ...     "sampler_process_num": 1,
    ...     "converter_process_num": 2,
    ...     "expand_factors": 10,
    ...     "strategies": "random",
    ... }
    >>> predictions = predict(model, graph, seed_ids, dataset_def)
    """
    output_signature = build_output_signature(
        graph.schema,
        dataset_def["meta_paths"],
        include_edge=dataset_def["include_edge"]
    )
    with EgoTensorGenerator(graph, **dataset_def) as data_gen:
        ds = tf.data.Dataset.from_generator(
            data_gen,
            args=(seed_ids,
                  dataset_def["batch_size"],
                  False,
                  # labels must be added even for prediction phase,
                  # since Model.predict ask dataset to return a tuple
                  # of either (`inputs`, `targets`) or (`inputs`, `targets`, `sample_weights`)
                  seed_ids
                  ),
            output_signature=output_signature
        )
        return seed_ids, softmax(model.predict(ds), axis=1)


def predict_and_explain(model, graph, seed_ids, dataset_def, explainer, target_class, prob_th):
    """
    Predict using `model` and explain result using `explainer` on `graph`.

    :param model: :class:`graph2tensor.model.models.MessagePassing` object, or other
                  equivalent object of subclass of `tf.keras.Model`
    :param graph: a :class:`graph2tensor.client.NumpyGraph` object to generate
                  dataset from
    :param seed_ids: ids of seed nodes in `np.1darray`
    :param dataset_def: dataset definition, see example below and
                        :class:`graph2tensor.model.data.DataGenerator` for more details
    :param explainer: explainer used to explain results
    :param target_class: index of target class in predictions to explain
    :param prob_th: the threshold of probability, only node with probability of `target_class`
                    greater than `prob_th` would be explained
    :return: the prediction results and its explanations

    :Example:
    >>> dataset_def = {
    ...     "meta_paths": ["(paper)-[cites]-(paper)-[cites]-(paper)", ],
    ...     "batch_size": 256,
    ...     "include_edge": False,
    ...     "sampler_process_num": 1,
    ...     "converter_process_num": 2,
    ...     "expand_factors": 10,
    ...     "strategies": "random",
    ... }
    >>> for explanations in predict_and_explain(model, graph, seed_ids, dataset_def, explainer, 0, .9):
    ...     pass
    """
    output_signature = build_output_signature(
        graph.schema,
        dataset_def["meta_paths"],
        include_edge=dataset_def["include_edge"]
    )
    vtypes = [re.findall('\((.+?)\)', path) for path in dataset_def['meta_paths']]
    etypes = [re.findall('\[(.+?)\]', path) for path in dataset_def['meta_paths']]
    with EgoTensorGenerator(graph, **dataset_def) as data_gen:
        ds = tf.data.Dataset.from_generator(
            data_gen,
            args=(seed_ids, dataset_def["batch_size"], False, seed_ids),
            output_signature=output_signature
        )
        for x, y in ds:
            grads, probs = explainer.explain(x, target_class)
            y, probs = y.numpy().tolist(), probs.numpy().tolist()
            for idx, (target_node, target_class_prob) in enumerate(zip(y, probs)):
                if target_class_prob < prob_th:
                    continue
                nodes = {}
                edges = []
                for ii, path in enumerate(grads):
                    current_src_idx = {idx}
                    next_src_idx = set()
                    for jj, (edge, edge_grads, segment_ids) in enumerate(path):
                        src_type, dst_type, etype = vtypes[ii][jj], vtypes[ii][jj+1], etypes[ii][jj]
                        max_seg_id = max(current_src_idx)
                        for kk, (src_id, dst_id, grad, seg_id) in enumerate(
                                zip(edge[UID].numpy().tolist(), edge[VID].numpy().tolist(),
                                edge_grads.numpy().tolist(), segment_ids.numpy().tolist())):
                            if seg_id > max_seg_id: break
                            if seg_id not in current_src_idx: continue
                            next_src_idx.add(kk)
                            nodes[src_id] = src_type
                            nodes[dst_id] = dst_type
                            edges.append(
                                {"src_id": src_id,
                                 "dst_id": dst_id,
                                 "etype": etype,
                                 "property": {"importance" : {"type": "number", "value": grad}}}
                            )
                        current_src_idx = next_src_idx
                        next_src_idx = set()
                data = {"vertexs": [{"id": k, "vtype": v, "property": {}} for k, v in nodes.items()], "edges": edges}
                yield target_node, target_class_prob, data

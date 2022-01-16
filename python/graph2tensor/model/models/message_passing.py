#!/usr/bin/env python3
import tensorflow as tf
from ..layers.attr_compact_layer import AttrCompact


class MessagePassing(tf.keras.Model):
    """
    A model framework that implemented message-passing programing paradigm.

    :param conv_layers: graph convolution layers, which receive a tuple
                        (src, edge, dst, offset) as inputs and output
                        aggregated messages from dst to each src。
                        Since there are usually several paths in sub-graph
                        and each path corresponding to a list of layers,
                        `conv_layers` should contain several lists of layers.
                        Different layer lists could be consisted of different layers
                        to handle different paths within ego-graph distinctively
    :param attr_reduce_mode: how to reduce attributes of nodes/edges, expected 'concat',
                             'mean' or 'sum', default to 'concat'
    :param pre_proc_layers: feature pre-processing layers, the first layer of it receive a
                            ego-graph tuple, see :class:`graph2tensor.converter.Converter`
                            for detail, the last layer outputs a ego-graph in which each
                            path is represented as a tuple of hops, (hop#1, hop#2, hop#3),
                            and each hop represented as (src, edge, dst, offset), in which
                            src, edge and dst are 2-d tensor and offset is 1-d tensor.
    :param reduce_layer: the layer reduce the message passed to centre nodes from each path
                            in ego-graph, if not specified, mean pooling will be applied
    :param out_layers: out layers, applied after convolution layers
    :param name: the name of model
    :param concat_hidden: whether concatenate the outputs of the last pre-processing layer
                            and every convolution layer
    :param kwargs: other args passed to super class
    """

    def __init__(self,
                 conv_layers,
                 attr_reduce_mode='concat',
                 pre_proc_layers=None,
                 reduce_layer=None,
                 out_layers=None,
                 name=None,
                 concat_hidden=False,
                 **kwargs):
        super(MessagePassing, self).__init__(name=name, **kwargs)
        self.conv_layers = conv_layers
        self.attr_reduce_layer = AttrCompact(mode=attr_reduce_mode, name='attr_reduce_layer')
        self.pre_proc_layers = pre_proc_layers
        self.reduce_layer = reduce_layer
        self.out_layers = out_layers
        self.concat_hidden = concat_hidden

    def call_pre_proc_layers(self, inputs, explaining=False):
        if self.pre_proc_layers is None:
            return inputs
        x = inputs
        for layer in self.pre_proc_layers:
            x = layer.explain_call(x) if explaining else layer(x)
        return x

    def call_attr_reduce_layer(self, inputs, explaining=False):
        return self.attr_reduce_layer.explain_call(inputs) if explaining \
            else self.attr_reduce_layer(inputs)

    def call_conv_layers(self, inputs, explaining=False):
        if len(self.conv_layers) != len(inputs):
            raise ValueError("path count in input do not match the convolution layers")
        outs = []
        for path, layers in zip(inputs, self.conv_layers):
            if len(path) != len(layers):
                raise ValueError("Hop count in path do not match the convolution layers")
            conv_out = []
            for layer in layers:
                msg_agg = []
                for hop in path:
                    msg_agg.append(layer.explain_call(hop) if explaining else layer(hop))
                conv_out.append(msg_agg[0])
                new_path = []
                for i in range(len(msg_agg)-1):
                    hop = [msg_agg[i], path[i][1], msg_agg[i+1], path[i][3]]
                    if explaining: hop.append(path[i][4])
                    new_path.append(tuple(hop))
                path = tuple(new_path)
            if self.concat_hidden:
                outs.append(tf.concat(conv_out, axis=1))
            else:
                outs.append(conv_out[-1])
        return outs

    def call_reduce_layer(self, inputs):
        if self.reduce_layer is None:
            return tf.reduce_mean(inputs, axis=0)
        return self.reduce_layer(inputs)

    def call_out_layers(self, inputs):
        if self.out_layers is None:
            return inputs
        x = inputs
        for layer in self.out_layers:
            x = layer(x)
        return x

    # The input tensors may have different shapes,
    # so set `experimental_relax_shapes` to `True`
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        x = self.call_pre_proc_layers(inputs)
        x = self.call_attr_reduce_layer(x)
        x = self.call_conv_layers(x)
        x = self.call_reduce_layer(x)
        x = self.call_out_layers(x)
        return x

    def explain_call(self, inputs):
        x = self.call_pre_proc_layers(inputs, True)
        x = self.call_attr_reduce_layer(x, True)
        x = self.call_conv_layers(x, True)
        x = self.call_reduce_layer(x)
        x = self.call_out_layers(x)
        return x

    def get_config(self):
        config = {
            "name": self.name,
            "concat_hidden": self.concat_hidden,
            "attr_reduce_mode": self.attr_reduce_layer.mode,
            "pre_proc_layers": None if self.pre_proc_layers is None else [tf.keras.layers.serialize(layer)
                                                                          for layer in self.pre_proc_layers],
            "reduce_layer": None if self.reduce_layer is None else tf.keras.layers.serialize(self.reduce_layer),
            "out_layers": None if self.out_layers is None else [tf.keras.layers.serialize(layer)
                                                                for layer in self.out_layers],
            "conv_layers": [
                [tf.keras.layers.serialize(layer) for layer in layers]
                for layers in self.conv_layers
            ]
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        pre_proc_layers = None if config["pre_proc_layers"] is None else [
            tf.keras.layers.deserialize(serialized_layer, custom_objects)
            for serialized_layer in config["pre_proc_layers"]
        ]
        conv_layers = [
            [tf.keras.layers.deserialize(serialized_layer, custom_objects) for serialized_layer in layers]
            for layers in config["conv_layers"]
        ]
        reduce_layer = None if config["reduce_layer"] is None else tf.keras.layers.deserialize(
            config["reduce_layer"], custom_objects)
        out_layers = None if config["out_layers"] is None else [
            tf.keras.layers.deserialize(serialized_layer, custom_objects)
            for serialized_layer in config["out_layers"]
        ]
        return cls(conv_layers, config["attr_reduce_mode"], pre_proc_layers,
                   reduce_layer, out_layers, config["name"], config["concat_hidden"])


if __name__ == "__main__":
    pass


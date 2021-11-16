# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
"""BYOM retinanet."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from byom.retinanet.layers.image_resize_layer import ImageResizeLayer


def _convert_to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def get_model_with_input(model, input_layer):
    """Implement a trick to replace input tensor."""
    _explored_layers = dict()
    for l in model.layers:
        _explored_layers[l.name] = [False, None]
    layers_to_explore = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
    model_outputs = {}
    # Loop until we reach the last layer.
    while layers_to_explore:
        layer = layers_to_explore.pop(0)
        # Skip layers that may be revisited in the graph to prevent duplicates.
        if not _explored_layers[layer.name][0]:
            # Check if all inbound layers explored for given layer.
            if not all([
                    _explored_layers[l.name][0]
                    for n in _convert_to_list(layer._inbound_nodes)
                    for l in _convert_to_list(n.inbound_layers)
                    ]):
                continue
            outputs = None
            # Visit input layer.
            if type(layer) == keras.layers.InputLayer:
                # skip input layer and use outside input tensors intead.
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = None
                layers_to_explore.extend([node.outbound_layer for
                                          node in layer._outbound_nodes])
                continue
            else:
                # Create new layer.
                layer_config = layer.get_config()
                # with keras.utils.CustomObjectScope({'PriorProbability': PriorProbability}):
                
                if type(layer) == ImageResizeLayer:
                    new_layer = ImageResizeLayer(output_dim=(1,1), output_scale=2)
                else:
                    new_layer = type(layer).from_config(layer_config)
                # Add to model.
                outputs = []
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    inbound_layers = node.inbound_layers
                    # For some reason, tf.keras does not always put things in a list.
                    if not isinstance(inbound_layers, list):
                        inbound_layers = [inbound_layers]
                    node_indices = [0] * len(inbound_layers)
                    for idx, l in enumerate(inbound_layers):
                        if type(l) == keras.layers.InputLayer:
                            prev_outputs.append(input_layer)
                        else:
                            keras_layer = _explored_layers[l.name][1]
                            if not isinstance(node_indices, list):
                                node_indices = [node_indices]
                            _tmp_output = keras_layer.get_output_at(node_indices[idx])
                            prev_outputs.append(_tmp_output)
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]

                    outputs.append(new_layer(prev_outputs))
                if len(outputs) == 1:
                    outputs = outputs[0]
                weights = layer.get_weights()
                if weights is not None and type(layer) != keras.layers.Dense:
                    try:
                        new_layer.set_weights(weights)
                    except ValueError:
                        print("{} is NOT loaded".format(layer.name))
            outbound_nodes = layer._outbound_nodes
            if not outbound_nodes:
                model_outputs[layer.output.name] = outputs
            layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
            # Mark current layer as visited and assign output nodes to the layer.
            _explored_layers[layer.name][0] = True
            _explored_layers[layer.name][1] = new_layer
        else:
            continue
    # Create new keras model object from pruned specifications.
    # only use input_image as Model Input.
    output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
    new_model = keras.models.Model(inputs=input_layer, outputs=output_tensors, name=model.name)

    return new_model


def retinanet(input_shape, model_path, inputs=None, training=True, input_name='Input'):
    """Build RetinaNet BYOM model."""
    if inputs is None:  
        new_input = tf.keras.Input(shape=input_shape,
                                   name=input_name)
    else:
        new_input = tf.keras.Input(tensor=inputs, name=input_name)
    loaded_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ImageResizeLayer': ImageResizeLayer})
    # loaded_model.summary()
    return get_model_with_input(loaded_model, new_input)

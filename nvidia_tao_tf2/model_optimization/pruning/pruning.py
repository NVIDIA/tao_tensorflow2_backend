# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TAO pruning.

This module includes APIs to prune a tf.keras model.
"""

import logging

import numpy as np
from tensorflow import keras

from nvidia_tao_tf2.backbones.utils_tf import swish
from nvidia_tao_tf2.common.decorators import override, subclass

from nvidia_tao_tf2.cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from nvidia_tao_tf2.cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion

"""Logger for pruning APIs."""
logger = logging.getLogger(__name__)


TRAVERSABLE_LAYERS = [
    keras.layers.BatchNormalization,
    keras.layers.Activation,
    keras.layers.Dropout,
    keras.layers.Softmax,
    keras.layers.MaxPooling2D,
    keras.layers.AveragePooling2D,
    keras.layers.Add,
    keras.layers.Subtract,
    keras.layers.Multiply,
    keras.layers.Average,
    keras.layers.Maximum,
    keras.layers.DepthwiseConv2D,
    keras.layers.ZeroPadding2D,
    keras.layers.ReLU,
    keras.layers.TimeDistributed,
    keras.layers.LeakyReLU,
    keras.layers.UpSampling2D,
    keras.layers.Conv2D,
    keras.layers.SeparableConv2D,
    ImageResizeLayer, WeightedFusion,
]


class Prune(object):
    """A class interface for the pruning operator."""

    def _get_filter_stats(self, kernels, layer):
        """Return norms of all kernel filters.

        Args:
            kernels (Array): array of kernels to get retained indices of, where the last
                dimension indexes individual kernels.
            layer(keras Layer): the layer whose filters we are going to make statistics.
        """
        raise NotImplementedError

    def _get_retained_idx(self, explored_stat):
        """Return indices of filters to retain.

        Args:
            explored_stat (1-d array): array of kernel filter norms.
        """
        raise NotImplementedError

    @override
    def prune(self, model):
        """Prune a model.

        Args:
            model (Model): the Keras model to prune.
        Returns:
            model (Model): the pruned model.
        """
        raise NotImplementedError()


class PrunedLayer(object):
    """Class definition to store information about pruned layers.

    Args:
        retained_idx (list): list of retained indices.
        output_tensor: Output tensor.
    """

    def __init__(
        self,
        retained_idx,
        explored_stat=None,
        visited=False,
        is_pruned=None,
        keras_layer=None,
    ):
        """Initialization routine.

        Args:
            retained_idx (list): All filter indices that are above the pruning threshold and may be
                retained.
            explored_stat (list): Norms of the filter in the given layer to keep track for full
                pruning of element-wise operations.
            is_pruned (bool): Flag to mark layer as pruned if num of retained filters is
                less than the number of incoming filters.
            visited (bool): Flag to mark layer as visited during phase 2 of the pruning.
            keras_layer (keras layer): Output of the current layer.
        Returns:
            Pruned_layer data structure.
        """
        self.retained_idx = retained_idx
        self.keras_layer = keras_layer
        self.is_pruned = is_pruned
        self.visited = visited
        self.explored_stat = explored_stat


@subclass
class PruneMinWeight(Prune):
    """A class that implements pruning according to the "min weight method".

    This function implements the 'min_weight' filtering method described on:
    [Molchanov et al.] Pruning Convolutional Neural Networks for Resource Efficient Inference,
    arXiv:1611.06440.

    For convolutional layers, only the norm of the kernels is considered (the norm of biases
    is ignored).

    Args:
        normalizer (str): 'max' to normalize by dividing each norm by the maximum norm within
            a layer; 'L2' to normalize by dividing by the L2 norm of the vector comprising all
            kernel norms.
        criterion (str): only 'L2' is supported.
        granularity (int): granularity of the number of filters to remove at a time.
        min_num_filters (int): minimum number of filters to retain in each layer.
        threshold (float): threshold to compare normalized norm against.
        equalization_criterion (str): criteria to equalize element wise operation layers.
            Supported criteria are 'arithmetic_mean', 'geometric_mean', 'union', 'intersection'.
        excluded_layers (list): list of names of layers that should not be pruned. Typical usage
            is for output layers of conv nets where the number of output channels must match
            a number of classes.
        byom_custom_layer (list): Option to specify BYOM custom layers. These layers will be
            included in the pass-through layers.
    """

    def __init__(
        self,
        normalizer,
        criterion,
        granularity,
        min_num_filters,
        threshold,
        excluded_layers=None,
        equalization_criterion="union",
        byom_custom_layer=None
    ):
        """Initialization routine."""
        self._normalizer = normalizer
        self._criterion = criterion
        self._granularity = granularity
        self._min_num_filters = min_num_filters
        self._equalization_criterion = equalization_criterion
        self._equalization_groups = []
        self._threshold = threshold
        if excluded_layers is None:
            excluded_layers = []
        self._excluded_layers = excluded_layers
        self._explored_layers = {}
        if byom_custom_layer is None:
            byom_custom_layer = []
        self.byom_custom_layers = byom_custom_layer
        self._visited = {}

    @staticmethod
    def _get_channel_index(data_format):
        """Return the channel index for the specified data format.

        Args:
            data_format (str): 'channels_first' or 'channels_last'.
        """
        if data_format == "channels_first":
            return -3
        if data_format == "channels_last":
            return -1
        raise ValueError(f"Unknown data format: {data_format}")

    def _get_data_format(self, layer):
        # Return a layer's data format. Recurse through previous layers
        # if necessary. If the data format cannot be determined, this
        # function returns ``None``.
        if hasattr(layer, "data_format"):
            return layer.data_format
        if isinstance(layer, keras.layers.TimeDistributed) and hasattr(layer.layer, 'data_format'):
            return layer.layer.data_format
        if type(layer) in [keras.layers.Reshape,
                           keras.layers.Permute] or \
            (isinstance(layer, keras.layers.TimeDistributed) and
             type(layer.layer) in [keras.layers.Reshape,
                                   keras.layers.Permute]):
            # Reshape and Permute layers make it impossible to retrieve the data format.
            return None
        if isinstance(layer, keras.layers.Flatten) or \
           (isinstance(layer, keras.layers.TimeDistributed) and
           isinstance(layer.layer, keras.layers.Flatten)):
            # Flatten layers yield (N, K) tensors and can be considered
            # either "channels_first" or "channels_last" indifferently.
            # Let's pick "channels_first" (N, C, H, W) arbitrarily.
            return "channels_first"
        inbound_layers = _get_inbound_layers(layer)
        if not inbound_layers:
            # We have not found the data format.
            return None
        # Recurse through inbound layers.
        data_formats = [self._get_data_format(inbound_l) for inbound_l in inbound_layers]

        if len(set(data_formats)) > 1:
            raise ValueError(
                f"Found more than 1 data format in inbound layers: {repr(data_formats)}"
            )
        return data_formats[0]

    def _get_previous_retained_idx(self, layer):
        inbound_layers = _get_inbound_layers(layer)
        if inbound_layers:
            return self._explored_layers[inbound_layers[0].name].retained_idx
        return None

    @override
    def _get_filter_stats(self, kernels, layer):
        """Return norms of all kernel filters.

        This function implements the 'min_weight' and returns the norms of the filters
        in the layer.

        Args:
            kernels (Array): array of kernels to get retained indices of, where the last
                dimension indexes individual kernels.
            layer(keras Layer): the layer whose filters we are going to make statistics.
        Returns:
            explored_stat (1-d array):  array of pruning stats for individual kernels
        """
        if self._criterion == "L2":
            pruning_stat = get_L2_norm(kernels, layer)
        else:
            raise NotImplementedError(f"{self._criterion} pruning")

        # Layer-wise normalization.
        pruning_stat = normalize_stat(pruning_stat, self._normalizer)

        return pruning_stat

    def _merge_layerwise_stats(self, layerwise_stats):
        """Merge the layerwise pruning stats according to equalization_criterion.

        Args:
            layerwise_stats (2-d array): Array of pruning stats for individual kernels
                in multiple weights.
        Returns:
            merged_stats (1-d array): Merged pruning stats.
        """
        if isinstance(layerwise_stats, list):
            layerwise_stats = np.stack(layerwise_stats)
        assert (
            layerwise_stats.ndim == 2
        ), "Layerwise stats has to be two dimensional array."

        if self._equalization_criterion == "union":
            # Equalizing input retained indices by computing union of the feature maps.
            merged_stat = np.amax(layerwise_stats, axis=0)
        elif self._equalization_criterion == "intersection":
            # Equalizing input retained indices by computing intersection of the
            # feature maps.
            merged_stat = np.amin(layerwise_stats, axis=0)
        elif self._equalization_criterion == "arithmetic_mean":
            # Equalizing input retained indices by making sure the mean of the filter norms
            # cross the threshold.
            merged_stat = np.mean(layerwise_stats, axis=0)
        elif self._equalization_criterion == "geometric_mean":
            # Equalizing input retained indices by making sure the geometric mean of the
            # filter norms cross the threshold.
            # Numpy handles np.exp(np.log(0.0)) == 0.0 OK, although throws a warning.
            log_stats = np.log(layerwise_stats)
            merged_stat = np.exp(np.mean(log_stats, axis=0))
        else:
            raise NotImplementedError(
                f"Unknown equalization criterion for element-wise operations: {self._equalization_criterion}"
            )

        return merged_stat

    @override
    def _get_retained_idx(self, explored_stat):
        """Return indices of filters to retain from explored stats (filter norms).

        This function computes the filter indices to be retained at the end of pruning.
        The number ouf filters is clamped to a multiple of the granularity.

        Args:
            explored_stat (1-d array): array of pruning stats for individual kernels
        Returns:
            retained_idx (1-d array): indices of filters to retain.
        """
        retained_idx = np.where(explored_stat > self._threshold)[0]

        # Compute depth of the layer before pruning.
        orig_output_depth = len(explored_stat)
        # Check granularity and minimum number of filters.
        num_retained = len(retained_idx)
        # Minimum number of filters - this shouldn't be more than the
        # original number of filters.
        min_num_filters = min(self._min_num_filters, orig_output_depth)
        # Maintaining atleast filters >= min_num_filters.
        num_retained = max(min_num_filters, num_retained)
        # Clamping to the nearest multiple of granularity.
        if num_retained % self._granularity > 0:
            num_retained += self._granularity - (num_retained % self._granularity)
        # Sanity check.
        num_retained = min(num_retained, orig_output_depth)
        # Sorting filter id's based on their pruning stat
        sorted_idx = np.argsort(-explored_stat)
        retained_idx = np.sort(sorted_idx[:num_retained])

        return retained_idx

    def _is_layer_pruned(self, layer):
        # Recurse through previous layers until we find one that is explicitly
        # pruned or not pruned.
        if layer.name in self._explored_layers:
            is_pruned = self._explored_layers[layer.name].is_pruned
            if is_pruned is not None:
                return is_pruned
        inbound_layers = _get_inbound_layers(layer)
        if inbound_layers:
            return any([self._is_layer_pruned(l) for l in inbound_layers])  # noqa pylint: disable=R1729
        return False

    def _equalize_retained_indices(self, eltwise_prunable_inputs):
        """Equalize retained indices of all inputs to the element wise layer."""
        if not isinstance(eltwise_prunable_inputs[0], keras.layers.TimeDistributed):
            output_depth = eltwise_prunable_inputs[0].filters
        else:
            output_depth = eltwise_prunable_inputs[0].layer.filters
        if any(layer.name in self._excluded_layers for layer in eltwise_prunable_inputs):
            logger.debug(
                "Skipping equalization of eltwise inputs: "  # noqa pylint: disable=C0209
                "{}".format(eltwise_prunable_inputs)
            )
            output_idx = range(output_depth)
        else:
            layerwise_stats = []
            for prunable_input in eltwise_prunable_inputs:
                layerwise_stats.append(
                    self._explored_layers[prunable_input.name].explored_stat
                )

            merged_stats = self._merge_layerwise_stats(layerwise_stats)

            output_idx = self._get_retained_idx(merged_stats)

        return output_idx

    def _equalize_dw_retained_indices(self,
                                      previous_stat,
                                      this_stat,
                                      previous_layer,
                                      this_layer,
                                      criterion):
        """Equalize the depth-wise conv. and its previous layer's retained indexes."""
        dw_layers = [previous_layer, this_layer]
        if isinstance(dw_layers[0], keras.layers.TimeDistributed):
            output_depth = dw_layers[0].layer.filters
        else:
            output_depth = dw_layers[0].filters
        if any((layer.name in self._excluded_layers or 'fpn_cell' in layer.name) for layer in dw_layers):
            logger.debug("Skipping equalization of depth-wise conv layers: {}".format(dw_layers))  # noqa pylint: disable=C0209
            output_idx = range(output_depth)
        else:
            cumulative_stat = previous_stat
            if criterion == "union":
                # Equalizing input retained indices by computing union of the feature maps.
                cumulative_stat = np.maximum(cumulative_stat, this_stat)
            elif criterion == "intersection":
                # Equalizing input retained indices by computing intersection of the
                # feature maps.
                cumulative_stat = np.minimum(cumulative_stat, this_stat)
            elif criterion == "arithmetic_mean":
                # Equalizing input retained indices by making sure the mean of the filter norms
                # cross the threshold.
                cumulative_stat = (cumulative_stat + this_stat) / 2.0
            elif criterion == "geometric_mean":
                # Equalizing input retained indices by making sure the geometric mean of the
                # filter norms cross the threshold.
                cumulative_stat = np.power(np.multiply(cumulative_stat,
                                           this_stat), float(1 / 2.0))
            else:
                raise NotImplementedError(
                    f"Unknown equalization criterion for depth-wise conv operations: {criterion}")

            # Clamp outputs to a multiple of the granularity
            output_idx = self._get_retained_idx(cumulative_stat)
        return output_idx

    def _explore_conv_transpose_layer(self, layer):
        # No transformation here, just propogate the number of output feature maps.
        kernels, _, _ = self._unravel_weights(layer)
        retained_idx = range(kernels.shape[2])
        return retained_idx

    def _explore_conv_or_fc_layer(self, layer):
        # Retrieve weights.
        kernels, _, _ = self._unravel_weights(layer)

        # Identify filters to prune.
        if layer.name in self._excluded_layers or 'fpn_cell' in layer.name:
            explored_stat = self._get_filter_stats(kernels, layer)
            retained_idx = range(kernels.shape[-1])
        else:
            explored_stat = self._get_filter_stats(kernels, layer)
            retained_idx = self._get_retained_idx(explored_stat)

        initial_neuron_count = kernels.shape[-1]
        retained_neuron_count = len(retained_idx)
        is_pruned = retained_neuron_count < initial_neuron_count

        return retained_idx, explored_stat, is_pruned

    def _explore_conv_dw_layer(self, layer):
        # Retrieve weights.
        kernels, _, _ = self._unravel_weights(layer)

        # Raise error when it's a DepthwiseConv2D layer but depth_multiplier != 1
        if kernels.shape[-1] != 1:
            raise ValueError('DepthwiseConv2D for pruning can only have depth_multiplier == 1.')

        # Identify filters to prune.
        if layer.name in self._excluded_layers or 'fpn_cell' in layer.name:
            explored_stat = self._get_filter_stats(kernels, layer)
            retained_idx = range(kernels.shape[2])
        else:
            explored_stat = self._get_filter_stats(kernels, layer)
            retained_idx = self._get_retained_idx(explored_stat)
        initial_neuron_count = kernels.shape[2]
        retained_neuron_count = len(retained_idx)
        # apply equalization for depth-wise conv.
        dw_layers = []
        dw_layers = find_prunable_parent(dw_layers,
                                         layer,
                                         True,
                                         visited=self._visited,
                                         byom_custom_layers=self.byom_custom_layers)
        self._update_equalization_groups(dw_layers + [layer])
        previous_layer = dw_layers[0]
        previous_stat = self._explored_layers[previous_layer.name].explored_stat
        retained_idx = self._equalize_dw_retained_indices(previous_stat,
                                                          explored_stat,
                                                          previous_layer,
                                                          layer,
                                                          self._equalization_criterion)
        retained_neuron_count = len(retained_idx)
        self._explored_layers[previous_layer.name].retained_idx = retained_idx
        is_pruned = retained_neuron_count < initial_neuron_count
        self._explored_layers[previous_layer.name].is_pruned = is_pruned

        return retained_idx, explored_stat, is_pruned

    def _explore_conv_2d_gru_layer(self, layer):
        # Retrieve weights:  W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h.
        weights = layer.get_weights()
        # Identify filters to prune.
        if layer.name in self._excluded_layers or not layer.trainable:
            if layer.name not in self._excluded_layers:
                logger.info("Skipping nontrainable layer: {}".format(layer.name))  # noqa pylint: disable=C0209
            # Infer output channels from first kernel.
            explored_stat = None
            retained_idx = range(weights[0].shape[-1])
        else:
            layerwise_stats = []
            # Do not take bias into account in the pruning decision,
            # use only W_z, W_r, W_h, U_z, U_r, U_h.
            for kernels in weights[:6]:
                layerwise_stats.append(self._get_filter_stats(kernels, layer))

            # Merge stats according to equalization criterion for determining joint pruned indices.
            # This handles the elementwise ops in the layer.
            explored_stat = self._merge_layerwise_stats(layerwise_stats)

            retained_idx = self._get_retained_idx(explored_stat)

        initial_neuron_count = weights[0].shape[-1]
        retained_neuron_count = len(retained_idx)
        is_pruned = retained_neuron_count < initial_neuron_count

        return retained_idx, explored_stat, is_pruned

    def _update_equalization_groups(self, eltwise_prunable_inputs):
        eq_groups = []
        for g in self._equalization_groups:
            for epi in eltwise_prunable_inputs:
                if epi in g:
                    eq_groups.append(g)
                    break
        merged_group = []
        for g in eq_groups:
            merged_group.extend(g)
        for epi in eltwise_prunable_inputs:
            if epi not in merged_group:
                merged_group.append(epi)
        for g in eq_groups:
            idx = self._equalization_groups.index(g)
            del self._equalization_groups[idx]
        self._equalization_groups.append(merged_group)
        return merged_group

    def _explore_elmtwise_layer(self, layer):
        eltwise_prunable_inputs = []
        eltwise_prunable_inputs = find_prunable_parent(
            eltwise_prunable_inputs, layer, byom_custom_layers=self.byom_custom_layers, visited=self._visited
        )
        logger.debug(
            "At explore_elmtwise_layer: Prunable parents at layer {}".format(layer.name)  # noqa pylint: disable=C0209
        )
        eltwise_prunable_inputs = list(set(eltwise_prunable_inputs))
        for l in eltwise_prunable_inputs:  # noqa pylint: disable=E741
            logger.debug("Prunable_parents {}".format(l.name))  # noqa pylint: disable=C0209
            # If any of the parents are broadcast layers, pop them out of prunable input list.
            if not isinstance(l, keras.layers.TimeDistributed) and l.filters == 1:
                # Set retained indices for this layer as 0.
                self._explored_layers[l.name].retained_idx = range(l.filters)
                self._explored_layers[l.name].is_pruned = False
                eltwise_prunable_inputs.pop(eltwise_prunable_inputs.index(l))
            elif isinstance(l, keras.layers.TimeDistributed) and l.layer.filters == 1:
                # Set retained indices for this layer as 0.
                self._explored_layers[l.name].retained_idx = range(l.layer.filters)
                self._explored_layers[l.name].is_pruned = False
                eltwise_prunable_inputs.pop(eltwise_prunable_inputs.index(l))
        # If newly updated eltwise true inputs have more than one branch, then
        # equalize the retained indices.
        if len(eltwise_prunable_inputs) > 1:
            if isinstance(layer, WeightedFusion):
                fixed_retained_idx = range(eltwise_prunable_inputs[0].filters)
            else:
                eltwise_prunable_inputs = self._update_equalization_groups(
                    eltwise_prunable_inputs
                )
                fixed_retained_idx = self._equalize_retained_indices(
                    eltwise_prunable_inputs
                )
        # Otherwise just prune the one conv layer as it was before.
        elif len(eltwise_prunable_inputs) == 1:
            layer_name = eltwise_prunable_inputs[-1].name
            fixed_retained_idx = self._explored_layers[layer_name].retained_idx
        else:
            # Retrieve weights.
            kernels, _, _ = self._unravel_weights(layer)
            return range(kernels.shape[-1])

        retained_idx = fixed_retained_idx
        # Set the newly calculated retained indices for all eltwise prunable layers while,
        # also checking if they were pruned during the equalization.
        for i in eltwise_prunable_inputs:
            self._explored_layers[i.name].retained_idx = retained_idx
            initial_neuron_count = i.get_weights()[0].shape[-1]
            pruned_state = len(retained_idx) < initial_neuron_count
            self._explored_layers[i.name].is_pruned = pruned_state
        # if the layer is a shared conv
        if isinstance(layer, keras.layers.Conv2D):
            logger.debug("Conv2D layer '{}' is shared.".format(layer.name))  # noqa pylint: disable=C0209
            retained_idx, _, _ = self._explore_conv_or_fc_layer(layer)

        return retained_idx

    def _explore_td_layer(self, layer):
        retained_idx = None
        explored_stat = None
        is_pruned = None
        if type(layer.layer) in [
            keras.layers.Conv2D,
            keras.layers.Dense
        ]:
            retained_idx, explored_stat, is_pruned = self._explore_conv_or_fc_layer(layer)
        elif type(layer.layer) in [keras.layers.DepthwiseConv2D]:
            retained_idx, explored_stat, is_pruned = self._explore_conv_dw_layer(layer)
        return retained_idx, explored_stat, is_pruned

    def _prune_explored_concat_layer(self, layer):
        data_format = self._get_data_format(layer)
        if data_format is not None:
            channel_index = self._get_channel_index(data_format)
            n_dims = len(layer.output_shape)
            allowed_axes = [channel_index % n_dims, channel_index % -n_dims]
            if layer.axis not in allowed_axes:
                raise ValueError(
                    f"Concatenation layer only supported on channel axis: data_format={data_format} axis={layer.axis}"
                )
        else:
            # The data format is unknown so we must make sure the previous layer was not pruned.
            if self._is_layer_pruned(layer):
                raise ValueError(
                    "Cannot process a concatenation layer if the data format "
                    "is unknown and one of the previous layers was pruned"
                )
            channel_index = layer.axis

        previous_layers = [l for n in layer._inbound_nodes for l in n.inbound_layers]  # noqa pylint: disable=E741
        retained_indices = []
        offset = 0
        for l in previous_layers:  # noqa pylint: disable=E741
            if self._is_layer_pruned(l):
                # Retain unpruned channels.
                retained_idx = self._explored_layers[l.name].retained_idx
            else:
                # Retain all channels.
                retained_idx = range(l.output_shape[channel_index])
            shifted_indices = [idx + offset for idx in retained_idx]
            retained_indices.extend(shifted_indices)
            offset += l.output_shape[channel_index]
        return retained_indices

    def _prune_explored_split_layer(self, layer):
        data_format = self._get_data_format(layer)
        if data_format is not None:
            channel_index = self._get_channel_index(data_format)
        else:
            channel_index = 1
        previous_layers = _get_inbound_layers(layer)
        pl = previous_layers[0]
        assert not self._is_layer_pruned(pl), (
            "Split layer's previous layer cannot be pruned. Try to add it "
            "to excluded layer list."
        )
        total_channels = pl.output_shape[channel_index]
        assert total_channels % layer.groups == 0, (
            "The number of channels of previous layer should be a multiple "
            "of the Split layer's group attribute."
        )
        n_channels = total_channels // layer.groups
        return range(n_channels)

    def _prune_explored_flatten_layer(self, layer):
        # We need to take the activations of previous layer into account.
        try:
            previous_layer = layer._inbound_nodes[-1].inbound_layers[0]
        except TypeError:
            # tf.keras could, for some weird reason, not make inbound nodes / layers
            # a list if there's only a single item for them.
            previous_layer = layer._inbound_nodes[-1].inbound_layers

        previous_layer_shape = previous_layer.output.get_shape()

        previous_data_format = self._get_data_format(previous_layer)
        previous_retained_idx = self._get_previous_retained_idx(layer)
        if previous_data_format in ['channels_first', 'channels_last']:
            assert previous_retained_idx is not None, ''
            'Previous retrained index of Flatten layer cannot be None if data format'
            ' is known.'
        if len(previous_layer_shape) != 4 and not isinstance(layer, keras.layers.TimeDistributed):
            raise ValueError(f"Expecting 4-dimensional activations got shape={repr(previous_layer_shape)}")
        # Take the spatial size into account and create a mask of activations to
        # retain from previous layer.
        if previous_data_format == "channels_first":
            # NCHW case.
            inp_spatial_size = int(np.prod(previous_layer_shape[-2:]))
            inp_num_fmaps = int(previous_layer_shape[-3])
            retained_filter_mask = np.asarray([False] * inp_num_fmaps)
            retained_filter_mask[previous_retained_idx] = True
            retained_activation_mask = np.repeat(retained_filter_mask, inp_spatial_size)
        elif previous_data_format == "channels_last":
            # NHWC case.
            inp_spatial_size = int(np.prod(previous_layer_shape[-3:-1]))
            inp_num_fmaps = int(previous_layer_shape[-1])
            retained_filter_mask = np.asarray([False] * inp_num_fmaps)
            retained_filter_mask[previous_retained_idx] = True
            retained_activation_mask = np.tile(retained_filter_mask, inp_spatial_size)
        elif previous_data_format is None:
            # The data format is unknown, make sure the previous layer was not pruned.
            retained_activation_mask = None
            if self._is_layer_pruned(layer):
                raise ValueError(
                    "Cannot process a pruned flatten layer if the "
                    "data format is unknown."
                )
        else:
            raise ValueError(f"Unknown data format: {previous_data_format}")
        if previous_data_format is not None:
            retained_idx = np.where(retained_activation_mask)[0]
        else:
            retained_idx = None
        return retained_idx

    def _prune_explored_batch_norm_layer(self, layer):
        # Propagate from previous layer.
        retained_idx = self._get_previous_retained_idx(layer)
        self._explored_layers[layer.name].retained_idx = retained_idx
        weights = tuple([w[retained_idx] for w in layer.get_weights()])  # noqa pylint: disable=R1728
        return weights

    def _prune_explored_conv_or_fc_layer(self, layer):
        # Retrieve weights.
        kernels, biases, scale_factor = self._unravel_weights(layer)
        previous_retained_idx = self._get_previous_retained_idx(layer)
        # Remove incoming connections that have been pruned in previous layer.
        is_conv2d = False
        if type(layer) in [keras.layers.Conv2D]:
            is_conv2d = True
        elif (isinstance(layer, keras.layers.TimeDistributed) and
              type(layer.layer) in [keras.layers.Conv2D]):
            is_conv2d = True
        if previous_retained_idx is not None:
            if is_conv2d:
                kernels = kernels[:, :, previous_retained_idx, :]
            else:
                kernels = kernels[previous_retained_idx, :]

        # Check if the current layer has been explored
        if layer.name not in self._explored_layers:
            raise ValueError(f"{layer.name} not explored")

        # Import retained idx from the explored stage.
        retained_idx = self._explored_layers[layer.name].retained_idx
        initial_neuron_count = kernels.shape[-1]
        retained_neuron_count = len(retained_idx)
        # Prune neurons from kernels and update layer spec.
        if is_conv2d:
            kernels = kernels[:, :, :, retained_idx]
            if type(layer) in [keras.layers.Conv2D]:
                layer.filters = retained_neuron_count
            else:
                layer.layer.filters = retained_neuron_count
        else:
            kernels = kernels[:, retained_idx]
            if type(layer) in [keras.layers.Dense]:
                layer.units = retained_neuron_count
            else:
                layer.layer.units = retained_neuron_count

        # Prune neurons from biases.
        if biases is not None:
            biases = biases[retained_idx]
            output_weights = (kernels, biases)
        else:
            output_weights = (kernels,)

        # Set scale factor for QuantizedConv2D layer.
        if scale_factor is not None:
            output_weights += (scale_factor,)

        msg = "layer %s: %d -> %d - actions: %s " % (  # noqa pylint: disable=C0209
            layer.name,
            initial_neuron_count,
            retained_neuron_count,
            "[name: %s]" % layer.name,  # noqa pylint: disable=C0209
        )
        logger.debug(msg)
        return output_weights

    def _prune_explored_conv_dw_layer(self, layer):
        # Retrieve weights.
        kernels, biases, scale_factor = self._unravel_weights(layer)
        initial_neuron_count = kernels.shape[2]
        # Check if the current layer has been explored
        if layer.name not in self._explored_layers:
            raise ValueError(f"{layer.name} not explored")

        # Import retained idx from the explored stage.
        retained_idx = self._explored_layers[layer.name].retained_idx
        kernels = kernels[:, :, retained_idx, :]
        retained_neuron_count = len(retained_idx)
        # Prune neurons from biases.
        if biases is not None:
            biases = biases[retained_idx]
            output_weights = (kernels, biases)
        else:
            output_weights = (kernels,)

        # Set scale factor for QuantizedDWConv layer.
        if scale_factor is not None:
            output_weights += (scale_factor,)

        msg = "layer %s: %d -> %d - actions: %s " % (  # noqa pylint: disable=C0209
            layer.name,
            initial_neuron_count,
            retained_neuron_count,
            "[name: %s]" % layer.name,  # noqa pylint: disable=C0209
        )
        logger.debug(msg)
        return output_weights

    def _prune_explored_conv_transpose_layer(self, layer):
        previous_retained_idx = self._get_previous_retained_idx(layer)
        kernels, biases, scale_factor = self._unravel_weights(layer)
        kernels = kernels[:, :, :, previous_retained_idx]
        weights = (kernels, biases) if biases is not None else (kernels,)
        # Set scale factor for QuantizedConvTranspose layer.
        if scale_factor is not None:
            weights += (scale_factor,)
        return weights

    def _prune_explored_conv_2d_gru_layer(self, layer):
        # Retrieve weights:  W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h
        weights = layer.get_weights()

        previous_retained_idx = self._get_previous_retained_idx(layer)

        # Remove incoming connections that have been pruned in previous layer.
        if previous_retained_idx is not None:
            # First three convlolution weights (W_z, W_r, W_h) operate on the input tensor.
            for idx, kernels in enumerate(weights[:3]):
                weights[idx] = kernels[:, :, previous_retained_idx, :]

        # Check if the current layer has been explored
        if layer.name not in self._explored_layers:
            raise ValueError(f"{layer.name} not explored")

        # Import retained idx from the explored stage.
        retained_idx = self._explored_layers[layer.name].retained_idx
        initial_neuron_count = weights[0].shape[-1]
        retained_neuron_count = len(retained_idx)

        # Remove incoming connections in the kernels that operate on the state (U_z, U_r, U_h).
        for idx, kernels in enumerate(weights[3:6]):
            weights[idx + 3] = kernels[:, :, retained_idx, :]

        # Prune output channels from all kernels (W_z, W_r, W_h, U_z, U_r, U_h).
        for idx, kernels in enumerate(weights[:6]):
            weights[idx] = kernels[:, :, :, retained_idx]

        # Prune output channels from biases (b_z, b_r, b_h).
        for idx, biases in enumerate(weights[6:]):
            weights[idx + 6] = biases[retained_idx]

        # Update layer config.
        layer.state_depth = retained_neuron_count
        layer.initial_state_shape = list(layer.initial_state_shape)
        layer.initial_state_shape[1] = retained_neuron_count

        msg = "layer %s: %d -> %d - actions: %s " % (  # noqa pylint: disable=C0209
            layer.name,
            initial_neuron_count,
            retained_neuron_count,
            "[name: %s]" % layer.name,  # noqa pylint: disable=C0209
        )
        logger.debug(msg)
        return weights

    def _prune_explored_td_layer(self, layer):
        weights = None
        retained_idx = None
        if type(layer.layer) in [
            keras.layers.Conv2D,
            keras.layers.Dense
        ]:
            weights = self._prune_explored_conv_or_fc_layer(layer)
        elif type(layer.layer) in [keras.layers.DepthwiseConv2D]:
            weights = self._prune_explored_conv_dw_layer(layer)
        elif type(layer.layer) in [keras.layers.BatchNormalization]:
            weights = self._prune_explored_batch_norm_layer(layer)
        elif isinstance(layer.layer, keras.layers.Flatten):
            retained_idx = self._prune_explored_flatten_layer(layer)
        elif isinstance(layer.layer, keras.layers.Concatenate):
            retained_idx = self._prune_explored_concat_layer(layer)
        else:
            retained_idx = self._get_previous_retained_idx(layer)
        return weights, retained_idx

    def _unravel_weights(self, layer):
        configs = layer.get_config()
        is_quantized = ('quantize' in configs) and configs['quantize']
        weights = layer.get_weights()
        if is_quantized:
            scaling_factor = weights[-1]
            weights = weights[:-1]
        else:
            scaling_factor = None

        if len(weights) == 1:
            kernels = weights[0]
            biases = None
        elif len(weights) == 2:
            kernels, biases = weights
        else:
            raise ValueError(f"Unhandled number of weights: {len(weights)}")
        return kernels, biases, scaling_factor

    def _convert_to_list(self, obj):
        if not isinstance(obj, list):
            return [obj]
        return obj

    def _explore(self, model):
        """Explore a model for pruning and decide the feature-maps for all layers.

        The model to prune must be a string of convolutional or fully-connected nodes. For example,
        the nv-Helnet family of models, the VGG-xx family of models, the ResNet-xx family of
        models, AlexNet or LeNet can be pruned using this API.

        This function implements the 'min_weight' filtering method described on:
        [Molchanov et al.] Pruning Convolutional Neural Networks for Resource Efficient Inference,
        arXiv:1611.06440.

        For convolutional layers, only the norm of the kernels is considered (the norm of biases
        is ignored).

        Args:
            model (Model): the Keras model to prune.
        Returns:
            model (Model): the explored model.
        """
        # Explore the model using Breadth First Traversal, starting from the input layers.

        logger.info("Exploring graph for retainable indices")
        layers_to_explore = model._input_layers
        model_inputs = []
        model_outputs = []

        # Loop until we reach the last layer.
        while layers_to_explore:

            layer = layers_to_explore.pop(0)
            logger.debug("Exploring layer : {}".format(layer.name))  # noqa pylint: disable=C0209

            go_to_another_layer = any([l.name not in self._explored_layers  # noqa pylint: disable=R1729
                                       for n in self._convert_to_list(layer._inbound_nodes)
                                       for l in self._convert_to_list(n.inbound_layers)])  # noqa pylint: disable=E741

            if go_to_another_layer:
                # Some of the inbound layers have not been explored yet.
                # Skip this layer for now, it will come back to the list
                # of layers to explore as the outbound layer of one of the
                # yet unexplored layers.
                continue

            retained_idx = None
            explored_stat = None
            outputs = None
            is_pruned = None

            # Layer-specific handling.
            if type(layer) in [
                    keras.layers.DepthwiseConv2D
            ]:
                retained_idx, explored_stat, is_pruned = self._explore_conv_dw_layer(layer)
            elif type(layer) in [
                    keras.layers.Conv2D,
                    keras.layers.Dense
            ]:
                elmtwise_inputs = _get_inbound_layers(layer)
                # Handle pruning for keras element wise merge layers.
                if len(layer._inbound_nodes) > 1 and len(set(elmtwise_inputs)) > 1:
                    # For eltwise layers check if all the inbound layers have been explored first.

                    if all(l.name in self._explored_layers for l in elmtwise_inputs):  # noqa pylint: disable=E741
                        retained_idx = self._explore_elmtwise_layer(layer)
                    else:
                        continue
                else:
                    # Explore current conv or fc layer for retainable feature maps.
                    retained_idx, explored_stat, is_pruned = self._explore_conv_or_fc_layer(
                        layer
                    )
            elif type(layer) in [
                    keras.layers.Conv2DTranspose
            ]:
                # Explore conv2d traspose layer for retainable indices.
                retained_idx = self._explore_conv_transpose_layer(layer)
            elif 'helper' in str(type(layer)) or isinstance(layer, keras.layers.Lambda):
                # @scha: BYOM custom layers are in format of <class 'helper.CustomLayer'>
                # Make sure that the previous layer was unpruned.
                pass
            elif type(layer) in [
                keras.layers.Activation,
                keras.layers.BatchNormalization,
                keras.layers.Dropout,
                keras.layers.MaxPooling2D,
                keras.layers.AveragePooling2D,
                keras.layers.GlobalAveragePooling2D,
                keras.layers.Softmax,
                keras.layers.ReLU,
                keras.layers.ELU,
                keras.layers.LeakyReLU,
                keras.layers.InputLayer,
                keras.layers.ZeroPadding2D,
                keras.layers.Flatten,
                keras.layers.Concatenate,
                keras.layers.UpSampling2D,
                keras.layers.Cropping2D,
                keras.models.Model,
                keras.layers.SeparableConv2D,
                ImageResizeLayer
            ]:
                # These layers are just pass-throughs.
                pass
                # TODO: type(layer) == keras.layers.SeparableConv2D:
                # _, pw_k, _ = layer.get_weights()
                # # Identify filters to prune.
                # if layer.name in self._excluded_layers:
                #     explored_stat = None
                #     retained_idx = range(pw_k.shape[-1])
                # else:
                #     explored_stat = self._get_filter_stats(pw_k, layer)
                #     retained_idx = self._get_retained_idx(explored_stat)
                # initial_neuron_count = pw_k.shape[-1]
                # retained_neuron_count = len(retained_idx)
                # is_pruned = retained_neuron_count < initial_neuron_count
            elif type(layer) in [keras.layers.Reshape, keras.layers.Permute]:
                # Make sure that the previous layer was unpruned.
                if self._is_layer_pruned(layer):
                    if (
                        isinstance(layer, keras.layers.Reshape) and
                        -1 in layer.target_shape
                    ):
                        retained_idx = None
                        is_pruned = None
                    else:
                        raise NotImplementedError(
                            "Reshape/Permute is not supported after a pruned layer."
                        )
                else:
                    retained_idx = None
                    is_pruned = False
            # Handle pruning for keras element wise merge layers.
            elif type(layer) in [
                keras.layers.Add,
                keras.layers.Subtract,
                keras.layers.Multiply,
                keras.layers.Average,
                keras.layers.Maximum,
                WeightedFusion
            ]:
                # For eltwise layers check if all the inbound layers have been explored first.
                elmtwise_inputs = [
                    l for n in layer._inbound_nodes for l in n.inbound_layers  # noqa pylint: disable=E741
                ]
                if all(l.name in self._explored_layers for l in elmtwise_inputs):  # noqa pylint: disable=E741
                    retained_idx = self._explore_elmtwise_layer(layer)
                else:
                    continue
            elif isinstance(layer, keras.layers.TimeDistributed):
                retained_idx, explored_stat, is_pruned = self._explore_td_layer(layer)
            elif isinstance(layer, keras.layers.Lambda):
                if layer.name not in self._excluded_layers:
                    raise ValueError(
                        "Lambda layers must be explicitly excluded from pruning. "  # noqa pylint: disable=C0209
                        "Met lambda layer with name {} that is not explicitly "
                        "excluded".format(layer.name)
                    )
                # Once we have verified that the lambda layer is excluded from pruning, it
                # can safely be assumed to be a pass-through.
                pass
            else:
                # Raise not implemented error for layers that aren't supported.
                raise NotImplementedError(f"Unknown layer type: {type(layer)}")

            # Explore the input layer.
            if type(layer) in [keras.layers.InputLayer]:
                # Re-use the existing InputLayer.
                outputs = layer.output
                model_inputs.append(outputs)
                retained_idx = None
            else:
                # Make sure there are no duplicates in the retained indices.
                if retained_idx is not None:
                    assert len(retained_idx) == len(set(retained_idx)), (
                        f"Duplicates found in list of retained indices: {repr(retained_idx)}"
                    )
            outbound_nodes = layer._outbound_nodes
            if not outbound_nodes:
                model_outputs.append(outputs)

            # @scha: in BYOM there are cases where the duplicate nodes are appended
            # As a result, the while loop in explore never ended
            for node in outbound_nodes:
                if node.outbound_layer not in layers_to_explore:
                    layers_to_explore.append(node.outbound_layer)

            names_to_explore = [l.name for l in layers_to_explore]  # noqa pylint: disable=E741
            logger.debug(
                "Updating layers to explore at {} to: {}".format(  # noqa pylint: disable=C0209
                    layer.name, names_to_explore  # noqa pylint: disable=C0209
                )
            )

            # Save info about the layer we just explored.
            self._explored_layers[layer.name] = PrunedLayer(
                retained_idx, explored_stat=explored_stat, is_pruned=is_pruned
            )
        return model

    @override
    def prune(self, model, layer_config_overrides=None, output_layers_with_outbound_nodes=None):
        """Prune an alreayd explored model, contains the retained-indices for all layers.

        The model to prune must be a string of convolutional or fully-connected nodes. For example,
        the nv-Helnet family of models, the VGG-xx family of models, the Resnet-xx family of
        models, AlexNet or LeNet can be pruned using this API.

        This function implements the 'min_weight' filtering method described on:
        [Molchanov et al.] Pruning Convolutional Neural Networks for Resource Efficient Inference,
        arXiv:1611.06440.

        For convolutional layers, only the norm of the kernels is considered (the norm of biases
        is ignored).

        Args:
            model (Model): the Keras model to prune.
            layer_config_overrides (dict): A dictionary of key-value pairs used for overriding
                layer configuration. Use cases include changing regularizers after pruning.
            output_layers_with_outbound_nodes (list): Option to specify intermediate output layers
                that have `outbound_nodes`.
        Returns:
            model (Model): the pruned model.
        """
        # get `training` config for BN reconstruction
        config_map = {l['name'] : l['inbound_nodes'] for l in model.get_config()['layers']}  # noqa pylint: disable=E741
        # Phase 1: Explore the model.
        if not output_layers_with_outbound_nodes:
            output_layers_with_outbound_nodes = []
        model = self._explore(model)

        # Phase 2: Prune the graph in Breadth First Search fashion, starting from the
        # input layer.
        logger.debug("Explored layers: {}".format(self._explored_layers.keys()))  # noqa pylint: disable=C0209
        logger.debug("Model layers: {}".format([l.name for l in model.layers]))  # noqa pylint: disable=C0209

        input_layer = [l for l in model.layers if (  # noqa pylint: disable=E741
            type(l) in [keras.layers.InputLayer])]
        layers_to_explore = input_layer
        model_outputs = {}
        logger.info("Pruning model and appending pruned nodes to new graph")

        # Loop until we reach the last layer.
        while layers_to_explore:

            layer = layers_to_explore.pop(0)
            # Skip layers that may be revisited in the graph to prevent duplicates.
            if not self._explored_layers[layer.name].visited:
                # Check if all inbound layers explored for given layer.

                inbound_nodes = layer._inbound_nodes
                # For some reason, tf.keras does not always put things in a list.
                if not isinstance(inbound_nodes, list):
                    inbound_nodes = [inbound_nodes]

                go_to_another_layer = False
                for n in inbound_nodes:
                    inbound_layers = n.inbound_layers
                    if not isinstance(inbound_layers, list):
                        inbound_layers = [inbound_layers]
                    for l in inbound_layers:  # noqa pylint: disable=E741
                        # if isinstance(l, dict):
                        #     break
                        if not self._explored_layers[l.name].visited:
                            go_to_another_layer = True
                            break

                    if go_to_another_layer:
                        break

                if go_to_another_layer:
                    # if not (len(prune_visited) == len(inbound_layers)):
                    # Some of the inbound layers have not gone through pruning phase yet.
                    # Skip this layer for now, it will come back to the list
                    # of layers to explore as the outbound layer of one of the
                    # yet unvisited layers.
                    continue

                logger.debug("Pruning layer: {}".format(layer.name))  # noqa pylint: disable=C0209
                weights = None
                outputs = None

                # Layer-specific handling. Carve weights out based on results from the
                # explore phase.
                if type(layer) in [
                        keras.layers.DepthwiseConv2D
                ]:
                    weights = self._prune_explored_conv_dw_layer(layer)
                elif type(layer) in [
                        keras.layers.Conv2D,
                        keras.layers.Dense
                ]:
                    weights = self._prune_explored_conv_or_fc_layer(layer)
                elif type(layer) in [keras.layers.BatchNormalization]:
                    weights = self._prune_explored_batch_norm_layer(layer)
                elif type(layer) in [
                        keras.layers.Conv2DTranspose
                ]:
                    weights = self._prune_explored_conv_transpose_layer(layer)
                elif isinstance(layer, keras.models.Model):
                    sub_model_layer = self.prune(layer)
                    weights = sub_model_layer.get_weights()
                elif (isinstance(layer, keras.layers.Concatenate)):
                    retained_idx = self._prune_explored_concat_layer(layer)
                    self._explored_layers[layer.name].retained_idx = retained_idx
                elif isinstance(layer, keras.layers.Flatten):
                    retained_idx = self._prune_explored_flatten_layer(layer)
                    self._explored_layers[layer.name].retained_idx = retained_idx
                elif type(layer) in [keras.layers.Reshape, keras.layers.Permute]:
                    # Make sure that the previous layer was unpruned.
                    if self._is_layer_pruned(layer):
                        if not (
                            isinstance(layer, keras.layers.Reshape) and
                            -1 in layer.target_shape
                        ):
                            raise NotImplementedError(
                                "Reshape is not supported after a pruned layer."
                            )
                    if (
                        isinstance(layer, keras.layers.Reshape) and
                        -1 in layer.target_shape
                    ):
                        retained_idx = self._get_previous_retained_idx(layer)
                        self._explored_layers[layer.name].retained_idx = retained_idx
                elif type(layer) in [
                    keras.layers.Activation,
                    keras.layers.Dropout,
                    keras.layers.MaxPooling2D,
                    keras.layers.AveragePooling2D,
                    keras.layers.GlobalAveragePooling2D,
                    keras.layers.Softmax,
                    keras.layers.ZeroPadding2D,
                    keras.layers.ReLU,
                    keras.layers.ELU,
                    keras.layers.LeakyReLU,
                    keras.layers.InputLayer,
                    keras.layers.Add,
                    keras.layers.Subtract,
                    keras.layers.Multiply,
                    keras.layers.Average,
                    keras.layers.Maximum,
                    keras.layers.UpSampling2D,
                    keras.layers.Cropping2D,
                    ImageResizeLayer,
                    WeightedFusion,
                    keras.layers.SeparableConv2D
                ]:
                    # These layers have no weights associated with them. Hence no transformation
                    # but, propogate retained indices from the previous layer.
                    retained_idx = self._get_previous_retained_idx(layer)
                    self._explored_layers[layer.name].retained_idx = retained_idx
                # elif type(layer) == keras.layers.SeparableConv2D:
                    # EXPERIMENTAL
                    # prev_retained_idx = self._get_previous_retained_idx(layer)
                    # self._explored_layers[layer.name].retained_idx = prev_retained_idx
                    # layer.filters = len(prev_retained_idx)
                    # TODO: address the issue from pruning the inner pw
                    # retained_idx = self._explored_layers[layer.name].retained_idx
                    # layer.filters = len(retained_idx)
                    # dw_k, pw_k, b = layer.get_weights()
                    # weights = [dw_k[:, :, prev_retained_idx, :],
                    #            pw_k[:, :, prev_retained_idx, :][..., retained_idx],
                    #            b[retained_idx]]
                    # raise NotImplementedError("Pruning doesn't support SeparableConv2D layer.")
                elif 'helper' in str(type(layer)) or isinstance(layer, keras.layers.Lambda):
                    # @scha: BYOM custom layers are in format of <class 'helper.CustomLayer'>
                    retained_idx = self._get_previous_retained_idx(layer)
                    self._explored_layers[layer.name].retained_idx = retained_idx
                elif type(layer) in [keras.layers.InputLayer]:
                    pass
                elif isinstance(layer, keras.layers.TimeDistributed):
                    weights, retained_idx = self._prune_explored_td_layer(layer)
                    if retained_idx is not None:
                        self._explored_layers[layer.name].retained_idx = retained_idx
                else:
                    # Other layers are not going through any transformation here.
                    raise NotImplementedError(
                        f"Unsupported layer type for layer names {layer.name} of type {type(layer)}"
                    )

                # Visit input layer.
                if type(layer) in [keras.layers.InputLayer]:
                    # Re-use the existing InputLayer.
                    outputs = layer.output
                    new_layer = layer
                else:
                    # Create new layer.
                    layer_config = layer.get_config()

                    # Apply layer config overrides.
                    if layer_config_overrides is not None:
                        for key in layer_config:
                            if key in layer_config_overrides:
                                layer_config[key] = layer_config_overrides[key]

                    with keras.utils.CustomObjectScope({'swish': swish}):
                        new_layer = type(layer).from_config(layer_config)

                    # Add to model.
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        inbound_layers = node.inbound_layers
                        # For some reason, tf.keras does not always put things in a list.
                        if not isinstance(inbound_layers, list):
                            inbound_layers = [inbound_layers]
                        node_indices = []
                        for node_i in range(len(node.keras_inputs)):
                            node_indices.append(
                                node.keras_inputs[node_i]._keras_history.node_index)
                        if len(node_indices) == 0:
                            node_indices = 0
                        # for idx, l in enumerate(node.inbound_layers):
                        for idx, l in enumerate(inbound_layers):

                            keras_layer = self._explored_layers[l.name].keras_layer
                            # TF2
                            # node_indices = node.node_indices
                            # For some reason, tf.keras does not always put things in a list.
                            if not isinstance(node_indices, list):
                                node_indices = [node_indices]

                            kl_output = keras_layer.get_output_at(node_indices[idx])

                            prev_outputs.append(kl_output)
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        if type(new_layer) in [keras.layers.BatchNormalization]:
                            if 'training' in config_map[layer.name][0][0][-1]:
                                outputs.append(
                                    new_layer(
                                        prev_outputs,
                                        training=config_map[layer.name][0][0][-1]['training'])
                                )
                            else:
                                outputs.append(new_layer(prev_outputs))
                        else:
                            outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    if weights is not None:
                        new_layer.set_weights(weights)
                outbound_nodes = layer._outbound_nodes
                if not outbound_nodes:
                    model_outputs[layer.output.name] = outputs
                # Option to specify intermediate output layers that have
                # have `outbound_nodes`
                if layer.name in output_layers_with_outbound_nodes:
                    model_outputs[layer.output.name] = outputs

                layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
                # Mark current layer as visited and assign output nodes to the layer.
                self._explored_layers[layer.name].visited = True
                self._explored_layers[layer.name].keras_layer = new_layer
            else:
                continue
        # Create new keras model object from pruned specifications.
        # Patch for duplicate outputs
        output_names = []
        for l in model.outputs:  # noqa pylint: disable=E741
            if l.name in model_outputs:
                if l.name not in output_names:
                    output_names.append(l.name)
        model_outputs = [model_outputs[name] for name in output_names]
        # model_outputs = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
        new_model = keras.models.Model(
            inputs=model.inputs, outputs=model_outputs, name=model.name
        )
        return new_model


def prune(
    model,
    method,
    normalizer,
    criterion,
    granularity,
    min_num_filters,
    threshold,
    excluded_layers=None,
    layer_config_overrides=None,
    equalization_criterion="union",
    output_layers_with_outbound_nodes=None,
    byom_custom_layer=None
):
    """Prune a model.

    The model to prune must be a Keras model, consisting of the following layer types:

    - 2D convolutions, transpose convolutions,
    - fully-connected,
    - batch normalization.

    The following non-parametric layer types are also supported:

    - non-linear activations (sigmoid, ReLU, TanH, ...),
    - flatten,
    - concatenation,
    - dropout,
    - element-wise operations (add, subtract, ...),
    - and more.

    For example, the nv-Helnet family of models, the VGG-xx family of models, the ResNet-xx family
    of models, AlexNet, LeNet or GoogleNet can be pruned using this API.

    The inbound layers to element-wise operations should not be pruned.

    This function implements the 'min_weight' filtering method described on:
    [Molchanov et al.] Pruning Convolutional Neural Networks for Resource Efficient Inference,
    arXiv:1611.06440.

    For convolutional layers, only the norm of the kernels is considered (the norm of biases
    is ignored).

    Args:
        model (Model): the Keras model to prune.
        method (str): only 'min_weight' is supported.
        normalizer (str): 'max' to normalize by dividing each norm by the maximum norm within
            a layer; 'L2' to normalize by dividing by the L2 norm of the vector comprising all
            kernel norms.
        criterion (str): only 'L2' is supported.
        granularity (int): granularity of the number of filters to remove at a time.
        min_num_filters (int): minimum number of filters to retain in each layer.
        threshold (float): threshold to compare normalized norm against.
        excluded_layers (list): list of names of layers that should not be pruned. Typical usage
            is for output layers of conv nets where the number of output channels must match
            a number of classes.
        layer_config_overrides (dict): A dictionary of key-value pairs used for overriding layer
            configuration. Use cases include changing regularizers after pruning.
        equalization_criterion (str): Criteria to equalize the stats of inputs to an element
            wise op layer. Options are [arithmetic_mean, geometric_mean, union, intersection].
        output_layers_with_outbound_nodes (list): Option to specify intermediate output layers
                that have `outbound_nodes`.
        byom_custom_layer (list): Option to specify BYOM custom layers. These layers will be
                pass-through.
    Returns:
        model (Model): the pruned model.
    """
    if excluded_layers is None:
        excluded_layers = []
    if method != "min_weight":
        # We don't know how to support other pruning methods.
        raise NotImplementedError(f"Unsupported pruning method: {method}")

    pruner = PruneMinWeight(
        normalizer,
        criterion,
        granularity,
        min_num_filters,
        threshold,
        equalization_criterion=equalization_criterion,
        excluded_layers=excluded_layers,
        byom_custom_layer=byom_custom_layer
    )

    return pruner.prune(model, layer_config_overrides, output_layers_with_outbound_nodes)


def _get_inbound_layers(layer):
    """Helper function to get the inbound layers of a given layer.

    Needed because tf.keras treats the `inbound_layers` / `inbound_nodes` attributes as single
    objects when there is only one of them, whereas keras treats them as lists regardless of how
    many elements they hold.

    Args:
        layer (keras.layers.Layer | tf.keras.layers.Layer): Layer for which to get inbound layers.

    Returns:
        inbound_layers (list): List of inbound layers.
    """
    inbound_layers = []
    inbound_nodes = layer._inbound_nodes
    if not isinstance(inbound_nodes, list):
        inbound_nodes = [inbound_nodes]
    for n in inbound_nodes:
        _inbound_layers = n.inbound_layers
        if not isinstance(_inbound_layers, list):
            _inbound_layers = [_inbound_layers]
        inbound_layers.extend(_inbound_layers)

    return inbound_layers


def normalize_stat(stat, normalizer):
    """Normalize pruning statistics.

    Args:
        stat (Array): array of statistics to normalize
        normalizer (str): either 'L2' (normalize by dividing by L2 norm) or
            'max' (normalize by dividing by max)
    Returns:
        The normalized array.
    """
    if normalizer == "L2":
        stat = stat / np.sqrt(np.sum(stat ** 2)) * len(stat)
    elif normalizer == "max":
        stat = stat / np.max(stat)
    elif normalizer != "off":
        raise NotImplementedError(f"Invalid pruning normalizer: {normalizer}")
    return stat


def get_L2_norm(kernels, layer):
    """Get the L2 norms of the filters for pruning.

    Args:
        kernels (Array): array of kernels to compute norms of, where the last
            dimension indexes individual kernels.
        layer(keras Layer): the layer whose filters we are going to make statistics.
            Special treatment to the DepthwiseConv2D.
    Returns:
        A vector of L2 norms, one for each kernel.
    """
    if type(layer) in [keras.layers.DepthwiseConv2D] or \
        (isinstance(layer, keras.layers.TimeDistributed) and
         type(layer.layer) in [keras.layers.DepthwiseConv2D]):
        # For DepthwiseConv2D, currently we only support depthwise_multiplier = 1.
        # I.e., axis 3 is always of size 1, kernel shape = (K_h, K_w, C_in, 1)
        norm = np.sqrt(np.sum(kernels**2, axis=(0, 1, 3)))
    else:
        norm = np.sqrt(np.sum(kernels**2, axis=tuple(range(kernels.ndim - 1))))

    return norm


def find_prunable_parent(prunable_parents,
                         layer,
                         skip_root=False,
                         visited=None,
                         byom_custom_layers=None):
    """Recursive function to find the first prunable parent in the current branch.

    Args:
        prunable_parents (list): A list of prunable parents accumulated till before the current
            layer was explored.
        layer (keras layer object): Current layer being explored.
        skip_root(bool): Whether or not to skip the root layer(the root of the recursion tree).
            This is useful for depthwise conv case, because the current layer is prunable,
            but we want to find its parent that is prunable rather than returning itself.
        byom_custom_layers (list): List of layers from BYOM to be added to TRAVERSABLE_LAYERS.
    Return:
        A list of keras layers which are prunable inputs to the given layer.
    """
    visited = visited or {}
    if byom_custom_layers:
        assert isinstance(byom_custom_layers, list), \
            f"Invalid data type for byom_custom_layers, {type(byom_custom_layers)}"
        TRAVERSABLE_LAYERS.extend(byom_custom_layers)

    # exit if you have encountered a prunable parent.
    if (type(layer) in [keras.layers.Conv2D,
                        keras.layers.Dense,
                        keras.layers.DepthwiseConv2D] and
        len(layer._inbound_nodes) == 1) or \
        (isinstance(layer, keras.layers.TimeDistributed) and
         type(layer.layer) in [keras.layers.Conv2D,
                               keras.layers.Dense,
                               keras.layers.DepthwiseConv2D]):
        if not skip_root:
            prunable_parents.extend([layer])
            return list(set(prunable_parents))
    # If you hit a shape manipulation layer, drop an exception.
    if type(layer) not in TRAVERSABLE_LAYERS:
        raise NotImplementedError(
            f"Pruning is not possible with {layer.name} layer " "in the way"
        )
    # Recurse across all branches to return prunable parents.
    previous_layers = []
    inbound_nodes = layer._inbound_nodes
    # For some reason, tf.keras does not always put things in a list.
    if not isinstance(inbound_nodes, list):
        inbound_nodes = [inbound_nodes]

    for n in inbound_nodes:
        inbound_layers = n.inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        for l in inbound_layers:  # noqa pylint: disable=E741
            previous_layers.append(l)

    for l in previous_layers:  # noqa pylint: disable=E741
        if visited and l.name in visited:
            prunable_parents.extend(visited[l.name])
        else:
            # Skip the Input layers if there are multiple parents.
            if type(l) not in [keras.layers.InputLayer]:
                find_prunable_parent(prunable_parents,
                                     l,
                                     False,
                                     visited,
                                     byom_custom_layers)
    visited[layer.name] = prunable_parents
    return list(set(prunable_parents))

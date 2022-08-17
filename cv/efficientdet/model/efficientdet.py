# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Keras implementation of efficientdet."""
import functools
from absl import logging
import numpy as np
import tensorflow as tf

from cv.efficientdet.model import model_builder
from cv.efficientdet.model import activation_builder
from cv.efficientdet.model import fpn_configs
from cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion
from cv.efficientdet.utils import hparams_config
from cv.efficientdet.utils import model_utils
from cv.efficientdet.utils import keras_utils
# pylint: disable=arguments-differ  # fo keras layers.


class SeparableConvWAR:
    """WAR for tf.keras.layers.SeparableConv2D."""

    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format=None,
            dilation_rate=(1, 1),
            depth_multiplier=1,
            activation=None,
            use_bias=True,
            depthwise_initializer='glorot_uniform',
            pointwise_initializer='glorot_uniform',
            bias_initializer='zeros',
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            pointwise_constraint=None,
            bias_constraint=None,
            name='separable_conv_war',
            **kwargs) -> None:
        """Init."""
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=False,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            name=name + '_dw',
            **kwargs)

        self.pw_conv = tf.keras.layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
            name=name + '_pw',
            **kwargs)

    def __call__(self, inputs):
        """Call."""
        x = self.dw_conv(inputs)
        x = self.pw_conv(x)
        return x


class FNode:
    """A Keras Layer implementing BiFPN Node."""

    def __init__(self,
                 feat_level,
                 inputs_offsets,
                 fpn_num_filters,
                 apply_bn_for_resampling,
                 is_training_bn,
                 conv_after_downsample,
                 conv_bn_act_pattern,
                 separable_conv,
                 act_type,
                 weight_method,
                 data_format,
                 name='fnode'):
        """Init Fnode."""
        self.feat_level = feat_level
        self.inputs_offsets = inputs_offsets
        self.fpn_num_filters = fpn_num_filters
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.separable_conv = separable_conv
        self.act_type = act_type
        self.is_training_bn = is_training_bn
        self.conv_after_downsample = conv_after_downsample
        self.data_format = data_format
        self.weight_method = weight_method
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.resample_layers = []
        self.name = name

        for i, input_offset in enumerate(self.inputs_offsets):
            self.resample_layers.append(
                ResampleFeatureMap(
                    self.feat_level,
                    self.fpn_num_filters,
                    self.apply_bn_for_resampling,
                    self.is_training_bn,
                    self.conv_after_downsample,
                    data_format=self.data_format,
                    name=self.name + f'resample_{i}_{input_offset}'))

        self.op_after_combine = OpAfterCombine(
            self.is_training_bn,
            self.conv_bn_act_pattern,
            self.separable_conv,
            self.fpn_num_filters,
            self.act_type,
            self.data_format,
            name=f'{self.name}_op_after_combine_{feat_level}')

        self.fuse_layer = WeightedFusion(
            inputs_offsets=self.inputs_offsets,
            name=f'fusion_{self.name}')

    def __call__(self, feats, training):
        """Call."""
        nodes = []
        for i, input_offset in enumerate(self.inputs_offsets):
            input_node = feats[input_offset]
            input_node = self.resample_layers[i](input_node, training, feats)
            nodes.append(input_node)
        new_node = self.fuse_layer(nodes)
        new_node = self.op_after_combine(new_node, training)
        return feats + [new_node]


class OpAfterCombine:
    """Operation after combining input features during feature fusion."""

    def __init__(self,
                 is_training_bn,
                 conv_bn_act_pattern,
                 separable_conv,
                 fpn_num_filters,
                 act_type,
                 data_format,
                 name='op_after_combine'):
        """Init OpAfterCombine."""
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.separable_conv = separable_conv
        self.fpn_num_filters = fpn_num_filters
        self.act_type = act_type
        self.data_format = data_format
        self.is_training_bn = is_training_bn
        if self.separable_conv:
            conv2d_layer = functools.partial(
                tf.keras.layers.SeparableConv2D, depth_multiplier=1)
            # conv2d_layer = functools.partial(
            #     SeparableConvWAR, depth_multiplier=1)
        else:
            conv2d_layer = tf.keras.layers.Conv2D

        self.conv_op = conv2d_layer(
            filters=fpn_num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=not self.conv_bn_act_pattern,
            data_format=self.data_format,
            name=name + '_conv')
        self.bn = keras_utils.build_batch_norm(
            is_training_bn=self.is_training_bn,
            data_format=self.data_format,
            name=name + '_bn')

    def __call__(self, new_node, training):
        """Call."""
        if not self.conv_bn_act_pattern:
            new_node = activation_builder.activation_fn(new_node, self.act_type)
        new_node = self.conv_op(new_node)
        new_node = self.bn(new_node, training=training)
        if self.conv_bn_act_pattern:
            new_node = activation_builder.activation_fn(new_node, self.act_type)
        return new_node


class ResampleFeatureMap:
    """Resample feature map for downsampling or upsampling."""

    def __init__(self,
                 feat_level,
                 target_num_channels,
                 apply_bn=False,
                 is_training_bn=None,
                 conv_after_downsample=False,
                 data_format=None,
                 pooling_type=None,
                 upsampling_type=None,
                 name='resample_p0'):
        """Init ResampleFeatureMap."""
        self.apply_bn = apply_bn
        self.is_training_bn = is_training_bn
        self.data_format = data_format
        self.target_num_channels = target_num_channels
        self.feat_level = feat_level
        self.conv_after_downsample = conv_after_downsample
        self.pooling_type = pooling_type or 'max'
        self.upsampling_type = upsampling_type or 'nearest'

        self.conv2d = tf.keras.layers.Conv2D(
            self.target_num_channels, (1, 1),
            padding='same',
            data_format=self.data_format,
            name=name + '_conv2d')
        self.bn = keras_utils.build_batch_norm(
            is_training_bn=self.is_training_bn,
            data_format=self.data_format,
            name=name + '_bn')

    def _pool2d(self, inputs, height, width, target_height, target_width):
        """Pool the inputs to target height and width."""
        height_stride_size = int((height - 1) // target_height + 1)
        width_stride_size = int((width - 1) // target_width + 1)
        if self.pooling_type == "max":
            model_class = tf.keras.layers.MaxPooling2D
        elif self.pooling_type == "avg":
            model_class = tf.keras.layers.AveragePooling2D
        else:
            raise NotImplementedError(f"Unsupported pooling type {self.pooling_type}")
        return model_class(
            pool_size=[height_stride_size + 1, width_stride_size + 1],
            strides=[height_stride_size, width_stride_size],
            padding='SAME',
            data_format=self.data_format)(inputs)

    def _maybe_apply_1x1(self, feat, training, num_channels):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != self.target_num_channels:
            feat = self.conv2d(feat)
            if self.apply_bn:
                feat = self.bn(feat, training=training)
        return feat

    def __call__(self, feat, training, all_feats):
        """Call."""
        hwc_idx = (2, 3, 1) if self.data_format == 'channels_first' else (1, 2, 3)
        height, width, num_channels = [feat.shape.as_list()[i] for i in hwc_idx]
        if all_feats:
            target_feat_shape = all_feats[self.feat_level].shape.as_list()
            target_height, target_width, _ = [target_feat_shape[i] for i in hwc_idx]
        else:
            # Default to downsampling if all_feats is empty.
            target_height, target_width = (height + 1) // 2, (width + 1) // 2

        # If conv_after_downsample is True, when downsampling, apply 1x1 after
        # downsampling for efficiency.
        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                feat = self._maybe_apply_1x1(feat, training, num_channels)
            feat = self._pool2d(feat, height, width, target_height, target_width)
            if self.conv_after_downsample:
                feat = self._maybe_apply_1x1(feat, training, num_channels)
        elif height <= target_height and width <= target_width:
            feat = self._maybe_apply_1x1(feat, training, num_channels)
            if height < target_height or width < target_width:
                feat = ImageResizeLayer(
                    target_height, target_width, self.data_format)(feat)
        else:
            raise ValueError(
                f'Incompatible Resampling : feat shape {height}x{width} '
                f'target_shape: {target_height}x{target_width}'
            )

        return feat


class ClassNet:
    """Object class prediction network."""

    def __init__(self,
                 num_classes=90,
                 num_anchors=9,
                 num_filters=32,
                 min_level=3,
                 max_level=7,
                 is_training_bn=False,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 data_format='channels_last',
                 name='class_net',
                 **kwargs):
        """Initialize the ClassNet.

        Args:
            num_classes: number of classes.
            num_anchors: number of anchors.
            num_filters: number of filters for "intermediate" layers.
            min_level: minimum level for features.
            max_level: maximum level for features.
            is_training_bn: True if we train the BatchNorm.
            act_type: String of the activation used.
            repeats: number of intermediate layers.
            separable_conv: True to use separable_conv instead of conv2D.
            survival_prob: if a value is set then drop connect will be used.
            data_format: string of 'channel_first' or 'channels_last'.
            name: the name of this layerl.
            **kwargs: other parameters.
        """
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_filters = num_filters
        self.min_level = min_level
        self.max_level = max_level
        self.repeats = repeats
        self.separable_conv = separable_conv
        self.is_training_bn = is_training_bn
        self.survival_prob = survival_prob
        self.act_type = act_type
        self.data_format = data_format
        self.conv_ops = []
        self.bns = []
        if separable_conv:
            conv2d_layer = functools.partial(
                tf.keras.layers.SeparableConv2D,
                # SeparableConvWAR,
                depth_multiplier=1,
                data_format=data_format,
                pointwise_initializer=tf.initializers.VarianceScaling(),
                depthwise_initializer=tf.initializers.VarianceScaling())
        else:
            conv2d_layer = functools.partial(
                tf.keras.layers.Conv2D,
                data_format=data_format,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        for i in range(self.repeats):
            # If using SeparableConv2D
            self.conv_ops.append(
                conv2d_layer(
                    self.num_filters,
                    kernel_size=3,
                    bias_initializer=tf.zeros_initializer(),
                    activation=None,
                    padding='same',
                    name=f'class-{i}'
                )
            )

            bn_per_level = []
            for level in range(self.min_level, self.max_level + 1):
                bn_per_level.append(
                    keras_utils.build_batch_norm(
                        is_training_bn=self.is_training_bn,
                        data_format=self.data_format,
                        name=f'class-{i}-bn-{level}',
                    )
                )
            self.bns.append(bn_per_level)

        self.classes = conv2d_layer(
            num_classes * num_anchors,
            kernel_size=3,
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            padding='same',
            name='class-predict')

    def __call__(self, inputs, training, **kwargs):
        """Call ClassNet."""
        class_outputs = []
        for level_id in range(0, self.max_level - self.min_level + 1):
            image = inputs[level_id]
            for i in range(self.repeats):
                original_image = image
                image = self.conv_ops[i](image)
                image = self.bns[i][level_id](image, training=training)
                if self.act_type:
                    image = activation_builder.activation_fn(image, self.act_type)
                if i > 0 and self.survival_prob:
                    image = model_utils.drop_connect(image, training, self.survival_prob)
                    image = image + original_image

            class_outputs.append(self.classes(image))

        return class_outputs


class BoxNet:
    """Box regression network."""

    def __init__(self,
                 num_anchors=9,
                 num_filters=32,
                 min_level=3,
                 max_level=7,
                 is_training_bn=False,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 data_format='channels_last',
                 name='box_net',
                 **kwargs):
        """Initialize BoxNet.

        Args:
            num_anchors: number of  anchors used.
            num_filters: number of filters for "intermediate" layers.
            min_level: minimum level for features.
            max_level: maximum level for features.
            is_training_bn: True if we train the BatchNorm.
            act_type: String of the activation used.
            repeats: number of "intermediate" layers.
            separable_conv: True to use separable_conv instead of conv2D.
            survival_prob: if a value is set then drop connect will be used.
            data_format: string of 'channel_first' or 'channels_last'.
            name: Name of the layer.
            **kwargs: other parameters.
        """
        self.num_anchors = num_anchors
        self.num_filters = num_filters
        self.min_level = min_level
        self.max_level = max_level
        self.repeats = repeats
        self.separable_conv = separable_conv
        self.is_training_bn = is_training_bn
        self.survival_prob = survival_prob
        self.act_type = act_type
        self.data_format = data_format

        self.conv_ops = []
        self.bns = []

        for i in range(self.repeats):
            # If using SeparableConv2D
            if self.separable_conv:
                self.conv_ops.append(
                    tf.keras.layers.SeparableConv2D(  # SeparableConvWAR(
                        filters=self.num_filters,
                        depth_multiplier=1,
                        pointwise_initializer=tf.initializers.VarianceScaling(),
                        depthwise_initializer=tf.initializers.VarianceScaling(),
                        data_format=self.data_format,
                        kernel_size=3,
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        padding='same',
                        name=f'box-{i}'))
            # If using Conv2d
            else:
                self.conv_ops.append(
                    tf.keras.layers.Conv2D(
                        filters=self.num_filters,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                        data_format=self.data_format,
                        kernel_size=3,
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        padding='same',
                        name=f'box-{i}'))

            bn_per_level = []
            for level in range(self.min_level, self.max_level + 1):
                bn_per_level.append(
                    keras_utils.build_batch_norm(
                        is_training_bn=self.is_training_bn,
                        data_format=self.data_format,
                        name=f'box-{i}-bn-{level}'
                    )
                )
            self.bns.append(bn_per_level)

        if self.separable_conv:
            self.boxes = tf.keras.layers.SeparableConv2D(  # SeparableConvWAR
                filters=4 * self.num_anchors,
                depth_multiplier=1,
                pointwise_initializer=tf.initializers.VarianceScaling(),
                depthwise_initializer=tf.initializers.VarianceScaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-predict')
        else:
            self.boxes = tf.keras.layers.Conv2D(
                filters=4 * self.num_anchors,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-predict')

    def __call__(self, inputs, training):
        """Call boxnet."""
        box_outputs = []
        for level_id in range(0, self.max_level - self.min_level + 1):
            image = inputs[level_id]
            for i in range(self.repeats):
                original_image = image
                image = self.conv_ops[i](image)
                image = self.bns[i][level_id](image, training=training)
                if self.act_type:
                    image = activation_builder.activation_fn(image, self.act_type)
                if i > 0 and self.survival_prob:
                    image = model_utils.drop_connect(image, training, self.survival_prob)
                    image = image + original_image

            box_outputs.append(self.boxes(image))

        return box_outputs


class SegmentationHead:
    """Keras layer for semantic segmentation head."""

    def __init__(self,
                 num_classes,
                 num_filters,
                 min_level,
                 max_level,
                 data_format,
                 is_training_bn,
                 act_type,
                 **kwargs):
        """Initialize SegmentationHead.

        Args:
            num_classes: number of classes.
            num_filters: number of filters for "intermediate" layers.
            min_level: minimum level for features.
            max_level: maximum level for features.
            data_format: string of 'channel_first' or 'channels_last'.
            is_training_bn: True if we train the BatchNorm.
            act_type: String of the activation used.
            **kwargs: other parameters.
        """
        self.act_type = act_type
        self.con2d_ts = []
        self.con2d_t_bns = []
        for _ in range(max_level - min_level):
            self.con2d_ts.append(
                tf.keras.layers.Conv2DTranspose(
                    num_filters,
                    3,
                    strides=2,
                    padding='same',
                    data_format=data_format,
                    use_bias=False))
            self.con2d_t_bns.append(
                keras_utils.build_batch_norm(
                    is_training_bn=is_training_bn,
                    data_format=data_format,
                    name='bn'))
        self.head_transpose = tf.keras.layers.Conv2DTranspose(
            num_classes, 3, strides=2, padding='same')

    def __call__(self, feats, training):
        """Call."""
        x = feats[-1]
        skips = list(reversed(feats[:-1]))

        for con2d_t, con2d_t_bn, skip in zip(self.con2d_ts, self.con2d_t_bns, skips):
            x = con2d_t(x)
            x = con2d_t_bn(x, training)
            x = activation_builder.activation_fn(x, self.act_type)
            x = tf.concat([x, skip], axis=-1)

        # This is the last layer of the model
        return self.head_transpose(x)  # 64x64 -> 128x128


class FPNCells:
    """FPN cells."""

    def __init__(self, config, name='fpn'):
        """Init FPNCells."""
        self.config = config

        if config.fpn_config:
            self.fpn_config = config.fpn_config
        else:
            self.fpn_config = fpn_configs.get_fpn_config(config.fpn_name,
                                                         config.min_level,
                                                         config.max_level,
                                                         config.fpn_weight_method)

        self.cells = [
            FPNCell(self.config, name=name + f'_cell_{rep}')
            for rep in range(self.config.fpn_cell_repeats)
        ]

    def __call__(self, feats, training):
        """Call."""
        for cell in self.cells:
            cell_feats = cell(feats, training)
            min_level = self.config.min_level
            max_level = self.config.max_level

            feats = []
            for level in range(min_level, max_level + 1):
                for i, fnode in enumerate(reversed(self.fpn_config.nodes)):
                    if fnode['feat_level'] == level:
                        feats.append(cell_feats[-1 - i])
                        break

        return feats


class FPNCell:
    """A single FPN cell."""

    def __init__(self, config, name='fpn_cell'):
        """Init FPNCell."""
        self.config = config
        if config.fpn_config:
            self.fpn_config = config.fpn_config
        else:
            self.fpn_config = fpn_configs.get_fpn_config(config.fpn_name,
                                                         config.min_level,
                                                         config.max_level,
                                                         config.fpn_weight_method)
        self.fnodes = []
        for i, fnode_cfg in enumerate(self.fpn_config.nodes):
            logging.debug(f'fnode {i} : {fnode_cfg}')
            fnode = FNode(
                fnode_cfg['feat_level'] - self.config.min_level,
                fnode_cfg['inputs_offsets'],
                config.fpn_num_filters,
                config.apply_bn_for_resampling,
                config.is_training_bn,
                config.conv_after_downsample,
                config.conv_bn_act_pattern,
                config.separable_conv,
                config.act_type,
                weight_method=self.fpn_config.weight_method,
                data_format=config.data_format,
                name=name + f'_fnode{i}')
            self.fnodes.append(fnode)

    def __call__(self, feats, training):
        """Call."""
        for fnode in self.fnodes:
            feats = fnode(feats, training)
        return feats


def efficientdet(input_shape, inputs=None, training=True, model_name=None, config=None):
    """Build EfficienDet model graph."""
    config = config or hparams_config.get_efficientdet_config(model_name)
    if inputs is None:
        inputs = tf.keras.Input(shape=input_shape)
    else:
        inputs = tf.keras.Input(tensor=inputs, shape=input_shape)

    # Feature network.
    resample_layers = []  # additional resampling layers.
    for level in range(6, config.max_level + 1):
        # Adds a coarser level by downsampling the last feature map.
        resample_layers.append(
            ResampleFeatureMap(
                feat_level=(level - config.min_level),
                target_num_channels=config.fpn_num_filters,
                apply_bn=config.apply_bn_for_resampling,
                is_training_bn=config.is_training_bn,
                conv_after_downsample=config.conv_after_downsample,
                data_format=config.data_format,
                name=f'resample_p{level}',
            ))
    fpn_cells = FPNCells(config)

    # class/box output prediction network.
    num_anchors = len(config.aspect_ratios) * config.num_scales
    num_filters = config.fpn_num_filters
    for head in config.heads:
        if head == 'object_detection':
            class_net = ClassNet(
                num_classes=config.num_classes,
                num_anchors=num_anchors,
                num_filters=num_filters,
                min_level=config.min_level,
                max_level=config.max_level,
                is_training_bn=config.is_training_bn,
                act_type=config.act_type,
                repeats=config.box_class_repeats,
                separable_conv=config.separable_conv,
                survival_prob=config.survival_prob,
                data_format=config.data_format)

            box_net = BoxNet(
                num_anchors=num_anchors,
                num_filters=num_filters,
                min_level=config.min_level,
                max_level=config.max_level,
                is_training_bn=config.is_training_bn,
                act_type=config.act_type,
                repeats=config.box_class_repeats,
                separable_conv=config.separable_conv,
                survival_prob=config.survival_prob,
                data_format=config.data_format)

        if head == 'segmentation':
            seg_head = SegmentationHead(
                num_classes=config.seg_num_classes,
                num_filters=num_filters,
                min_level=config.min_level,
                max_level=config.max_level,
                is_training_bn=config.is_training_bn,
                act_type=config.act_type,
                data_format=config.data_format)

    # call backbone network.
    all_feats = model_builder.build_backbone(inputs, config)
    # TODO(@yuw): wrap line
    feats = [all_feats[k] for k in sorted(all_feats.keys())][config.min_level:config.max_level + 1]
    # feats = all_feats[config.min_level:config.max_level + 1]

    # Build additional input features that are not from backbone.
    for resample_layer in resample_layers:
        feats.append(resample_layer(feats[-1], training, None))

    # call feature network.
    fpn_feats = fpn_cells(feats, training)

    # call class/box/seg output network.
    outputs = []
    if 'object_detection' in config.heads:
        class_outputs = class_net(fpn_feats, training)
        box_outputs = box_net(fpn_feats, training)
        outputs.extend([class_outputs, box_outputs])
    if 'segmentation' in config.heads:
        seg_outputs = seg_head(fpn_feats, training)
        outputs.append(seg_outputs)

    final_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.name)
    return final_model

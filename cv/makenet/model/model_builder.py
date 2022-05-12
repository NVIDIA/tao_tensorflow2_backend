# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO classification model builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.models import Model

from backbones.efficientnet_tf import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7
)
from backbones.resnet_tf import ResNet
from backbones.mobilenet_tf import MobileNet, MobileNetV2
from cv.makenet.utils.helper import decode_tltb


SUPPORTED_ARCHS = [
    "resnet", "efficientnet_b0", "efficientnet_b1",
    "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5",
    "efficientnet_b6", "efficientnet_b7",
    "mobilenet_v1", "mobilenet_v2", "byom"
]


def add_dense_head(nclasses, base_model, data_format, kernel_regularizer, bias_regularizer):
    """Wrapper to add dense head to the backbone structure."""
    output = base_model.output
    output_shape = output.get_shape().as_list()
    if data_format == 'channels_first':
        pool_size = (output_shape[-2], output_shape[-1])
    else:
        pool_size = (output_shape[-3], output_shape[-2])
    output = AveragePooling2D(pool_size=pool_size, name='avg_pool',
                              data_format=data_format, padding='valid')(output)
    output = Flatten(name='flatten')(output)
    # updated per TF2 documentation (https://www.tensorflow.org/guide/mixed_precision)
    output = Dense(nclasses, name='predictions_dense',
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(output)
    outputs = Activation('softmax', dtype='float32', name='predictions')(output)
    final_model = Model(inputs=base_model.input, outputs=output, name=base_model.name)
    return final_model


def get_resnet(nlayers=18,
               input_shape=(3, 224, 224),
               data_format='channels_first',
               nclasses=1000,
               kernel_regularizer=None,
               bias_regularizer=None,
               all_projections=True,
               use_batch_norm=True,
               use_pooling=False,
               retain_head=False,
               use_bias=True,
               freeze_bn=False,
               freeze_blocks=None):
    """Wrapper to get ResNet backbone."""
    input_image = Input(shape=input_shape)
    final_model = ResNet(nlayers=nlayers,
                         input_tensor=input_image,
                         data_format=data_format,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         use_batch_norm=use_batch_norm,
                         activation_type='relu',
                         all_projections=all_projections,
                         use_pooling=use_pooling,
                         add_head=retain_head,
                         nclasses=nclasses,
                         freeze_blocks=freeze_blocks,
                         freeze_bn=freeze_bn,
                         use_bias=use_bias)
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_byom(model_config_path=None,
             input_shape=(3, 224, 224),
             freeze_blocks=None,
             passphrase=None,
             data_format='channels_first',
             nclasses=1000,
             kernel_regularizer=None,
             bias_regularizer=None,
             retain_head=False):
    """Wrapper to get Bring Your Own Model from json file."""

    # For BYOM, we don't have the code for the model archicture.
    # As a result, we must load from eff file
    out_dict = decode_tltb(model_config_path, passphrase=passphrase)
    final_model = out_dict['model']

    # Rename the model to be more meaningful
    final_model._name = out_dict['model_name']

    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_efficientnet_b0(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B0 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB0(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b1(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B1 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB1(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b2(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B2 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB2(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b3(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B3 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB3(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b4(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B4 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB4(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b5(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B5 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB5(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b6(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B6 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB6(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b7(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B7 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB7(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_mobilenet(input_shape=None,
                  data_format='channels_first',
                  nclasses=1000,
                  use_batch_norm=None,
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  retain_head=False,
                  use_bias=True,
                  freeze_bn=False,
                  freeze_blocks=None,
                  stride=32):
    """Wrapper to get MobileNet model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = MobileNet(inputs=input_image,
                            input_shape=input_shape,
                            dropout=0.0,
                            add_head=retain_head,
                            stride=stride,
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            nclasses=nclasses,
                            use_batch_norm=use_batch_norm,
                            use_bias=use_bias,
                            freeze_bn=freeze_bn,
                            freeze_blocks=freeze_blocks)
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_mobilenet_v2(input_shape=None,
                     data_format='channels_first',
                     nclasses=1000,
                     use_batch_norm=None,
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     retain_head=False,
                     all_projections=False,
                     use_bias=True,
                     freeze_bn=False,
                     freeze_blocks=None,
                     stride=32):
    """Wrapper to get MobileNet V2 model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = MobileNetV2(inputs=input_image,
                              input_shape=input_shape,
                              add_head=retain_head,
                              stride=stride,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              all_projections=all_projections,
                              nclasses=nclasses,
                              use_batch_norm=use_batch_norm,
                              use_bias=use_bias,
                              freeze_bn=freeze_bn,
                              freeze_blocks=freeze_blocks)

    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


# defining model dictionary
model_choose = {"resnet": get_resnet,
                "efficientnet_b0": get_efficientnet_b0,
                "efficientnet_b1": get_efficientnet_b1,
                "efficientnet_b2": get_efficientnet_b2,
                "efficientnet_b3": get_efficientnet_b3,
                "efficientnet_b4": get_efficientnet_b4,
                "efficientnet_b5": get_efficientnet_b5,
                "efficientnet_b6": get_efficientnet_b6,
                "efficientnet_b7": get_efficientnet_b7,
                "mobilenet_v1": get_mobilenet,
                "mobilenet_v2": get_mobilenet_v2,
                "byom": get_byom}


def get_model(arch="resnet",
              input_shape=(3, 224, 224),
              data_format=None,
              nclasses=1000,
              kernel_regularizer=None,
              bias_regularizer=None,
              retain_head=False,
              freeze_blocks=None,
              **kwargs):
    """Wrapper to choose feature extractor given backbone name."""

    kwa = dict()
    if arch == 'resnet':
        kwa['nlayers'] = kwargs['nlayers']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_pooling'] = kwargs['use_pooling']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['all_projections'] = kwargs['all_projections']
    elif 'efficientnet_b' in arch:
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = None # TODO(@yuw): kwargs['activation'].activation_type
    elif arch == 'mobilenet_v1':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
    elif arch == 'mobilenet_v2':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['all_projections'] = kwargs['all_projections']
    elif arch == 'byom':
        kwa['model_config_path'] = kwargs['model_config_path']
        kwa['passphrase'] = kwargs['passphrase'] 
    else:
        raise ValueError('Unsupported architecture: {}'.format(arch))

    model = model_choose[arch](input_shape=input_shape,
                               nclasses=nclasses,
                               data_format=data_format,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               retain_head=retain_head,
                               freeze_blocks=freeze_blocks,
                               **kwa)
    return model

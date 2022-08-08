# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Classification training script with protobuf configuration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import json
import logging
import os

from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase, ResNetV1QDQCase
from tensorflow_quantization.quantize import quantize_model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from PIL import Image, ImageFile

import horovod.tensorflow.keras as hvd
# Horovod: initialize Horovod.
hvd.init()

from cv.classification.callback.eff_checkpoint import EffCheckpoint
from cv.classification.config.hydra_runner import hydra_runner
from cv.classification.config.default_config import ExperimentConfig
from cv.classification.model.model_builder import get_model
from cv.classification.utils.mixup_generator import MixupImageDataGenerator
from cv.classification.utils.preprocess_input import preprocess_input
from cv.classification.utils import preprocess_crop  # noqa pylint: disable=unused-import
from cv.classification.utils.helper import (
    build_lr_scheduler,
    build_optimizer,
    decode_tltb,
    load_model,
    setup_config)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 9000000000
logger = logging.getLogger(__name__)
verbose = 0


def setup_callbacks(ckpt_freq, results_dir, lr_config,
                    init_epoch, iters_per_epoch, max_epoch, key,
                    hvd):
    """Setup callbacks: tensorboard, checkpointer, lrscheduler, csvlogger.

    Args:
        ckpt_freq (int): checkpoint and validation frequency.
        results_dir (str): Path to a folder where various training outputs will
                           be written.
        init_epoch (int): The number of epoch to resume training.
        key: encryption key
        hvd: horovod instance
    Returns:
        callbacks (list of keras.callbacks): list of callbacks.
    """
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                hvd.callbacks.MetricAverageCallback()]
    max_iterations = iters_per_epoch * max_epoch
    lrscheduler = build_lr_scheduler(lr_config, hvd.size(), max_iterations)
    init_step = (init_epoch - 1) * iters_per_epoch
    lrscheduler.reset(init_step)
    callbacks.append(lrscheduler)

    if hvd.rank() == 0:
        # Set up the checkpointer.
        save_weights_dir = os.path.join(results_dir, 'weights')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        if not os.path.exists(save_weights_dir):
            os.makedirs(save_weights_dir)
        # Save encrypted models
        checkpointer = EffCheckpoint(save_weights_dir, key, verbose=0, ckpt_freq=ckpt_freq)
        callbacks.append(checkpointer)

        # Set up the custom TensorBoard callback. It will log the loss
        # after every step, and some images and user-set summaries only on
        # the first step of every epoch.
        tensorboard = TensorBoard(log_dir=results_dir)
        callbacks.append(tensorboard)

        # Set up the CSV logger, logging statistics after every epoch.
        csvfilename = os.path.join(results_dir, 'training.csv')
        csvlogger = CSVLogger(csvfilename,
                              separator=',',
                              append=False)
        callbacks.append(csvlogger)

    return callbacks


def load_data(train_data, val_data, preprocessing_func,
              image_height, image_width, batch_size,
              enable_random_crop=False, enable_center_crop=False,
              enable_color_augmentation=False,
              interpolation=0, color_mode="rgb",
              mixup_alpha=0.0, no_horizontal_flip=False, data_format='channels_first'):
    """Load training and validation data with default data augmentation.

    Args:
        train_data (str): path to the training data.
        val_data (str): path to the validation data.
        preprocessing_func: function to process an image.
        image_height (int): Height of the input image tensor.
        image_width (int): Width of the input image tensor.
        batch_size (int): Number of image tensors per batch.
        enable_random_crop (bool): Flag to enable random cropping in load_img.
        enable_center_crop (bool): Flag to enable center cropping for val.
        interpolation(int): Interpolation method for image resize. 0 means bilinear,
            while 1 means bicubic.
        color_mode (str): Input image read mode as either `rgb` or `grayscale`.
        mixup_alpha (float): mixup alpha.
        no_horizontal_flip(bool): Flag to disable horizontal flip for
            direction-aware datasets.
    Return:
        train/val Iterators and number of classes in the dataset.
    """
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    preprocess_crop._set_color_augmentation(enable_color_augmentation)
    interpolation = interpolation_map[interpolation]
    # WARNING!!!
    # Deprecated: `tf.keras.preprocessing.image.ImageDataGenerator` is not
    # recommended for new code. Prefer loading images with
    # `tf.keras.utils.image_dataset_from_directory` and transforming the output
    # `tf.data.Dataset` with preprocessing layers. For more information, see the
    # tutorials for [loading images](
    # https://www.tensorflow.org/tutorials/load_data/images) and
    # [augmenting images](
    # https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    # the [preprocessing layer guide](
    # https://www.tensorflow.org/guide/keras/preprocessing_layers).

    # set color augmentation properly for train.
    # this global var will not affect validation dataset because
    # the crop method is either "none" or "center" for val dataset,
    # while this color augmentation is only possible for "random" crop.
    # Initializing data generator : Train
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=not no_horizontal_flip,
        featurewise_center=False,
        data_format=data_format)

    train_iterator = MixupImageDataGenerator(
        train_datagen, train_data, batch_size,
        image_height, image_width,
        color_mode=color_mode,
        interpolation=interpolation + ':random' if enable_random_crop else interpolation,
        alpha=mixup_alpha
    )
    logger.info('Processing dataset (train): {}'.format(train_data))

    # Initializing data generator: Val
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=False,
        data_format=data_format)

    # Initializing data iterator: Val
    val_iterator = val_datagen.flow_from_directory(
        val_data,
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=batch_size,
        interpolation=interpolation + ':center' if enable_center_crop else interpolation,
        shuffle=False,
        class_mode='categorical')
    logger.info(f'Processing dataset (validation): {val_data}')

    # Check if the number of classes is > 1
    assert train_iterator.num_classes > 1, \
        "Number of classes should be greater than 1. Consider adding a background class."

    # Check if the number of classes is consistent
    assert train_iterator.num_classes == val_iterator.num_classes, \
        "Number of classes in train and val don't match."
    return train_iterator, val_iterator, train_iterator.num_classes


def run_experiment(cfg, results_dir=None,
                   key=None, init_epoch=1, verbosity=False):
    """Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster
          submission works.

    Args:
        config_path (str): Path to a text file containing a complete experiment
                           configuration.
        results_dir (str): Path to a folder where various training outputs will
                           be written.
        If the folder does not already exist, it will be created.
        init_epoch (int): The number of epoch to resume training.
    """
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    verbose = 1 if hvd.rank() == 0 else 0
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level='DEBUG' if verbosity else 'INFO')

    nchannels, image_height, image_width = cfg['model_config']['input_image_size']
    assert nchannels in [1, 3], "Invalid input image dimension."
    assert image_height >= 16, "Image height should be greater than 15 pixels."
    assert image_width >= 16, "Image width should be greater than 15 pixels."

    if nchannels == 3:
        color_mode = "rgb"
    else:
        color_mode = "grayscale"

    # Load augmented data
    train_iterator, val_iterator, nclasses = \
        load_data(cfg['train_config']['train_dataset_path'],
                  cfg['train_config']['val_dataset_path'],
                  partial(preprocess_input,
                          data_format=cfg['data_format'],
                          mode=cfg['train_config']['preprocess_mode'],
                          img_mean=list(cfg['train_config']['image_mean']),
                          color_mode=color_mode),
                  image_height, image_width,
                  batch_size=cfg['train_config']['batch_size_per_gpu'],
                  enable_random_crop=cfg['train_config']['enable_random_crop'],
                  enable_center_crop=cfg['train_config']['enable_center_crop'],
                  enable_color_augmentation=cfg['train_config']['enable_color_augmentation'],
                  color_mode=color_mode,
                  mixup_alpha=cfg['train_config']['mixup_alpha'],
                  no_horizontal_flip=cfg['train_config']['disable_horizontal_flip'],
                  data_format=cfg['data_format'])

    # @scha: For BYOM model loading
    if cfg['model_config']['arch'] in ["byom"] and cfg['model_config']['byom_model'] == '':
        raise ValueError('{} requires .tltb file to be processed by TAO'.format(cfg['model_config']['arch']))

    ka = dict(
        nlayers=cfg['model_config']['n_layers'],
        use_batch_norm=cfg['model_config']['use_batch_norm'],
        use_pooling=cfg['model_config']['use_pooling'],
        freeze_bn=cfg['model_config']['freeze_bn'],
        use_bias = cfg['model_config']['use_bias'],
        all_projections=cfg['model_config']['all_projections'],
        dropout=cfg['model_config']['dropout'],
        model_config_path = cfg['model_config']['byom_model'],
        passphrase = cfg['key']
    )
    input_shape = (nchannels, image_height, image_width) \
        if cfg['data_format'] == 'channels_first' else (image_height, image_width, nchannels)

    final_model = get_model(
        arch=cfg['model_config']['arch'],
        input_shape=input_shape,
        data_format=cfg['data_format'],
        nclasses=nclasses,
        retain_head=cfg['model_config']['retain_head'],
        freeze_blocks=cfg['model_config']['freeze_blocks'],
        **ka)

    # @scha: Load CUSTOM_OBJS from BYOM
    if cfg['model_config']['arch'] in ["byom"]:
        custom_objs = decode_tltb(ka['model_config_path'], ka['passphrase'])['custom_objs']
    else:
        custom_objs = {}

    # Set up BN and regularizer config
    bn_config = None
    reg_config = cfg['train_config']['reg_config']
    final_model = setup_config(
        final_model,
        reg_config,
        bn_config=bn_config,
        custom_objs=custom_objs
    )

    if cfg['train_config']['pretrained_model_path']:
        # Decrypt and load pretrained model
        pretrained_model = load_model(
            cfg['train_config']['pretrained_model_path'],
            passphrase=cfg['key'])

        strict_mode = True
        for layer in pretrained_model.layers[1:]:
            # The layer must match up to prediction layers.
            if layer.name == 'predictions_dense':
                strict_mode = False
            try:
                l_return = final_model.get_layer(layer.name)
            except ValueError:
                # Some layers are not there
                continue
            try:
                l_return.set_weights(layer.get_weights())
            except ValueError:
                if strict_mode:
                    # This is a pruned model
                    final_model = setup_config(
                        pretrained_model,
                        reg_config,
                        bn_config=bn_config
                    )
    if cfg['train_config']['qat']:
        qdq_cases = [EfficientNetQDQCase(), ResNetV1QDQCase()] \
            if 'efficientnet' in cfg['model_config']['arch'] else [ResNetV1QDQCase()]
        final_model = quantize_model(final_model, custom_qdq_cases=qdq_cases)

    # Printing model summary
    if hvd.rank() == 0:
        final_model.summary()

    if cfg['init_epoch'] > 1 and not cfg['train_config']['pretrained_model_path']:
        raise ValueError("Make sure to load the correct model when setting initial epoch > 1.")

    if cfg['train_config']['pretrained_model_path'] and cfg['init_epoch'] > 1:
        final_model = pretrained_model
        opt = pretrained_model.optimizer
    else:
        # Defining optimizer
        opt = build_optimizer(cfg['train_config']['optim_config'])
    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True)
    # Compiling model
    cc = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=cfg['train_config']['label_smoothing'])
    final_model.compile(
        loss=cc,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
        optimizer=opt,
        experimental_run_tf_function=False)

    # Setup callbacks
    callbacks = setup_callbacks(cfg['train_config']['checkpoint_freq'],
                                results_dir,
                                cfg['train_config']['lr_config'],
                                init_epoch, len(train_iterator) // hvd.size(),
                                cfg['train_config']['n_epochs'], key,
                                hvd)
    # Writing out class-map file for inference mapping
    if hvd.rank() == 0:
        with open(os.path.join(results_dir, "classmap.json"), "w", encoding='utf-8') as f:
            json.dump(train_iterator.class_indices, f)

    # Commencing Training
    final_model.fit(
        train_iterator,
        steps_per_epoch=len(train_iterator) // hvd.size(),
        epochs=cfg['train_config']['n_epochs'],
        verbose=verbose,
        workers=cfg['train_config']['n_workers'],
        validation_data=val_iterator,
        validation_steps=len(val_iterator), # // hvd.size(),
        validation_freq=cfg['train_config']['checkpoint_freq'],
        callbacks=callbacks,
        initial_epoch=init_epoch - 1)

    print("Training finished successfully.")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application.
    """
    run_experiment(cfg=cfg,
                   results_dir=cfg.results_dir,
                   key=cfg.key,
                   init_epoch=cfg.init_epoch)
    logger.info("Training finished successfully.")


if __name__ == '__main__':
    main()

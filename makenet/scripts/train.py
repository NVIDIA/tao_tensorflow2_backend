# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

"""Makenet training script with protobuf configuration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import logging
import os


import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from PIL import ImageFile
import six

import horovod.tensorflow.keras as hvd
# Horovod: initialize Horovod.
hvd.init()

from makenet.config.hydra_runner import hydra_runner
from makenet.config.default_config import ExperimentConfig
from common.utils import SoftStartCosineAnnealingScheduler
from makenet.model.model_builder import get_model
from makenet.utils.mixup_generator import MixupImageDataGenerator
from makenet.utils.preprocess_input import preprocess_input
from makenet.utils import preprocess_crop  # noqa pylint: disable=unused-import
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
verbose = 0


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def setup_callbacks(model_name, results_dir,
                    init_epoch, iters_per_epoch, max_epoch, key,
                    hvd):
    """Setup callbacks: tensorboard, checkpointer, lrscheduler, csvlogger.

    Args:
        model_name (str): name of the model used.
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
    lrscheduler = SoftStartCosineAnnealingScheduler(
        base_lr=0.05 * hvd.size(),
        min_lr_ratio=0.02,
        soft_start=0.1,
        max_iterations=max_iterations
    )
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
        weight_filename = os.path.join(save_weights_dir,
                                       '%s_{epoch:03d}.hdf5' % model_name)
        checkpointer = ModelCheckpoint(weight_filename, key, verbose=1)
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
              interpolation=0, color_mode="rgb",
              no_horizontal_flip=False):
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
        no_horizontal_flip(bool): Flag to disable horizontal flip for
            direction-aware datasets.
    Return:
        train/val Iterators and number of classes in the dataset.
    """
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation = interpolation_map[interpolation]
    # set color augmentation properly for train.
    # this global var will not affect validation dataset because
    # the crop method is either "none" or "center" for val dataset,
    # while this color augmentation is only possible for "random" crop.
    # Initializing data generator : Train
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=not no_horizontal_flip,
        featurewise_center=False)
    train_iterator = train_datagen.flow_from_directory(
        val_data,
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=batch_size,
        interpolation=interpolation + ':center' if enable_random_crop else interpolation,
        shuffle=False,
        class_mode='categorical')
    logger.info('Processing dataset (train): {}'.format(train_data))

    # Initializing data generator: Val
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=False)

    # Initializing data iterator: Val
    val_iterator = val_datagen.flow_from_directory(
        val_data,
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=batch_size,
        interpolation=interpolation + ':center' if enable_center_crop else interpolation,
        shuffle=False,
        class_mode='categorical')
    logger.info('Processing dataset (validation): {}'.format(val_data))

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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # check if model parallelism is enabled or not
    world_size = 1
    gpus = list(range(hvd.local_rank() * world_size, (hvd.local_rank() + 1) * world_size))
    config.gpu_options.visible_device_list = ','.join([str(x) for x in gpus])
    K.set_session(tf.compat.v1.Session(config=config))
    verbose = 1 if hvd.rank() == 0 else 0
    K.set_image_data_format('channels_first')
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level='DEBUG' if verbosity else 'INFO')

    # Set random seed.
    logger.debug("Random seed is set to {}".format(cfg['train_config']['random_seed']))
    # Configure tf logger verbosity.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # TODO: get channel, height and width of the input image
    nchannels, image_height, image_width = 3, 224, 224
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
                          data_format='channels_first',
                          mode=cfg['train_config']['preprocess_mode'],
                          img_mean=None,
                          color_mode=color_mode),
                  image_height, image_width,
                  batch_size_per_gpu=cfg['train_config']['batch_size_per_gpu'],
                  enable_random_crop=cfg['train_config']['enable_random_crop'],
                  enable_center_crop=cfg['train_config']['enable_center_crop'],
                  color_mode=color_mode,
                  no_horizontal_flip=cfg['train_config']['disable_horizontal_flip'])

    # TODO: Creating model
    ka = dict()
    ka['nlayers'] =  18
    ka['use_batch_norm'] = True
    ka['use_pooling'] = True
    ka['freeze_bn'] = False
    ka['use_bias'] = False
    ka['all_projections'] = False
    ka['dropout'] = 0.0
    freeze_blocks = None
    # TODO
    final_model = get_model(arch='resnet',
                            input_shape=(nchannels, image_height, image_width),
                            data_format='channels_first',
                            nclasses=nclasses,
                            use_imagenet_head=False,
                            freeze_blocks=freeze_blocks,
                            **ka)

    # Printing model summary
    final_model.summary()

    if cfg['init_epoch'] > 1 and not cfg['train_config']['pretrained_model_path']:
        raise ValueError("Make sure to load the correct model when setting initial epoch > 1.")

    if cfg['train_config']['pretrained_model_path'] and cfg['init_epoch'] > 1:
        raise NotImplementedError
    else:
        # Defining optimizer
        opt = tf.keras.optimizers.SGD(
            lr=0.05,
            momentum=0.9,
            nesterov=True
        )
    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)
    # Compiling model
    cc = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=cfg['train_config']['label_smoothing'])
    final_model.compile(loss=cc, metrics=['accuracy'],
                        optimizer=opt)

    # Setup callbacks
    callbacks = setup_callbacks('resnet', results_dir,
                                init_epoch, len(train_iterator) // hvd.size(),
                                cfg['train_config']['n_epochs'], key,
                                hvd)
    # Writing out class-map file for inference mapping
    if hvd.rank() == 0:
        with open(os.path.join(results_dir, "classmap.json"), "w") \
             as classdump:
            json.dump(train_iterator.class_indices, classdump)

    # Commencing Training
    final_model.fit_generator(
        train_iterator,
        steps_per_epoch=len(train_iterator) // hvd.size(),
        epochs=cfg['train_config']['n_epochs'],
        verbose=verbose,
        workers=cfg['train_config']['n_workers'],
        validation_data=val_iterator,
        validation_steps=len(val_iterator),
        callbacks=callbacks,
        initial_epoch=init_epoch - 1)

    # Evaluate the model on the full data set.
    score = hvd.allreduce(
        final_model.evaluate_generator(val_iterator,
                                       len(val_iterator),
                                       workers=cfg['train_config']['n_workers']))
    if verbose:
        logger.info('Total Val Loss: {}'.format(score[0]))
        logger.info('Total Val accuracy: {}'.format(score[1]))


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of MakeNet application.
    """
    run_experiment(cfg=cfg,
                   results_dir=cfg.results_dir,
                   key=cfg.key,
                   init_epoch=cfg.init_epoch)
    logger.info("Training finished successfully.")


if __name__ == '__main__':
    main()

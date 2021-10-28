# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
"""Perform Makenet Evaluation on IVA car make dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import logging

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from PIL import ImageFile
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from iva.common.utils import check_tf_oom
from iva.makenet.spec_handling.spec_loader import load_experiment_spec
from iva.makenet.utils.helper import get_input_shape, model_io, setup_config
from iva.makenet.utils import preprocess_crop  # noqa pylint: disable=unused-import
from iva.makenet.utils.preprocess_input import preprocess_input

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
K.set_image_data_format('channels_first')

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='Evaluate a classification model.')
    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        required=True,
        type=str,
        help='Path to the experiment spec file..'
    )
    parser.add_argument(
        '-k',
        '--key',
        required=True,
        type=str,
        help='Key to load a .tlt model.'
    )
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


def run_evaluate(args=None):
    """Wrapper function to run evaluation of MakeNet model.

    Args:
       Dictionary arguments containing parameters parsed in the main function.
    """
    # Set up logger verbosity.
    verbosity = 'INFO'
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=verbosity)

    # Configure tf logger verbosity.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load experiment spec.
    if args.experiment_spec_file is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", args.experiment_spec_file)
        # The spec in config_path has to be complete.
        # Default spec is not merged into es.
        es = load_experiment_spec(args.experiment_spec_file,
                                  merge_from_default=False,
                                  validation_schema="validation")
    else:
        logger.info("Loading the default experiment spec.")
        es = load_experiment_spec(validation_schema="validation")

    # Decrypt and load the pretrained model
    final_model = model_io(es.eval_config.model_path, enc_key=args.key)
    # override BN config
    if es.model_config.HasField("batch_norm_config"):
        bn_config = es.model_config.batch_norm_config
    else:
        bn_config = None
    # reg_config and freeze_bn are actually not useful, just use bn_config
    # so the BN layer's output produces correct result.
    # of course, only the BN epsilon matters in evaluation.
    final_model = setup_config(
        final_model,
        es.train_config.reg_config,
        freeze_bn=es.model_config.freeze_bn,
        bn_config=bn_config
    )
    # Defining optimizer
    opt = keras.optimizers.SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=False)
    # Define precision/recall and F score metrics
    topk_acc = partial(keras.metrics.top_k_categorical_accuracy,
                       k=es.eval_config.top_k)
    topk_acc.__name__ = 'topk_acc'
    # Compile model
    final_model.compile(loss='categorical_crossentropy',
                        metrics=[topk_acc],
                        optimizer=opt)

    # print model summary
    final_model.summary()

    # Get input shape
    image_height, image_width, nchannels = get_input_shape(final_model)

    assert nchannels in [1, 3], (
        "Unsupported channel count {} for evaluation".format(nchannels)
    )
    color_mode = "rgb"
    if nchannels == 1:
        color_mode = "grayscale"
    interpolation = es.model_config.resize_interpolation_method
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation = interpolation_map[interpolation]
    if es.eval_config.enable_center_crop:
        interpolation += ":center"

    img_mean = es.train_config.image_mean
    if nchannels == 3:
        if img_mean:
            assert all(c in img_mean for c in ['r', 'g', 'b']) , (
                "'r', 'g', 'b' should all be present in image_mean "
                "for images with 3 channels."
            )
            img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
        else:
            img_mean = [103.939, 116.779, 123.68]
    else:
        if img_mean:
            assert 'l' in img_mean, (
                "'l' should be present in image_mean for images "
                "with 1 channel."
            )
            img_mean = [img_mean['l']]
        else:
            img_mean = [117.3786]

    # Initializing data generator
    target_datagen = ImageDataGenerator(
        preprocessing_function=partial(preprocess_input,
                                       data_format='channels_first',
                                       mode=es.train_config.preprocess_mode,
                                       img_mean=img_mean,
                                       color_mode=color_mode),
        horizontal_flip=False)
    # Initializing data iterator
    target_iterator = target_datagen.flow_from_directory(
        es.eval_config.eval_dataset_path,
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=es.eval_config.batch_size,
        class_mode='categorical',
        interpolation=interpolation,
        shuffle=False)
    logger.info('Processing dataset (evaluation): {}'.format(es.eval_config.eval_dataset_path))
    nclasses = target_iterator.num_classes
    assert nclasses > 1, "Invalid number of classes in the evaluation dataset."

    # If number of classes does not match the new data
    assert nclasses == final_model.output.get_shape().as_list()[-1], \
        "The number of classes of the loaded model doesn't match the \
         number of classes in the evaluation dataset."

    # Evaluate the model on the full data set.
    score = final_model.evaluate_generator(target_iterator,
                                           len(target_iterator),
                                           workers=es.eval_config.n_workers,
                                           use_multiprocessing=False)

    print('Evaluation Loss: {}'.format(score[0]))
    print('Evaluation Top K accuracy: {}'.format(score[1]))
    # Re-initializing data iterator
    target_iterator = target_datagen.flow_from_directory(
        es.eval_config.eval_dataset_path,
        target_size=(image_height, image_width),
        batch_size=es.eval_config.batch_size,
        color_mode=color_mode,
        class_mode='categorical',
        interpolation=interpolation,
        shuffle=False)
    logger.info("Calculating per-class P/R and confusion matrix. It may take a while...")
    Y_pred = final_model.predict_generator(target_iterator, len(target_iterator), workers=1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(target_iterator.classes, y_pred))
    print('Classification Report')
    class_dict = target_iterator.class_indices
    target_names = [c[0] for c in sorted(class_dict.items(), key=lambda x:x[1])]
    print(classification_report(target_iterator.classes, y_pred, target_names=target_names))


@check_tf_oom
def main(args=None):
    """Wrapper function for evaluating MakeNet application.

    Args:
       args: Dictionary arguments containing parameters defined by command line
             parameters.
    """
    # parse command line
    args = parse_command_line(args)
    run_evaluate(args)


if __name__ == '__main__':
    main()

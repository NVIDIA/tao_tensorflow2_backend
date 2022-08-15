# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
"""Perform classification evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial
import logging
import zipfile

import numpy as np
from PIL import ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from eff.core import Archive

from common.hydra.hydra_runner import hydra_runner

from cv.classification.config.default_config import ExperimentConfig
from cv.classification.utils import preprocess_crop  # noqa pylint: disable=unused-import
from cv.classification.utils.preprocess_input import preprocess_input
from cv.classification.utils.helper import initialize, get_input_shape, load_model
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


def run_evaluate(cfg):
    """Wrapper function to run evaluation of classification model.

    Args:
       Dictionary arguments containing parameters parsed in the main function.
    """
    # Set up logger verbosity.
    verbosity = 'INFO'
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=verbosity)
    # set backend
    initialize()
    # Decrypt EFF
    final_model = load_model(
        str(cfg['eval_config']['model_path']),
        cfg['key'])

    # Defining optimizer
    opt = keras.optimizers.SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=False)
    # Define precision/recall and F score metrics
    topk_acc = partial(keras.metrics.top_k_categorical_accuracy,
                       k=cfg['eval_config']['top_k'])
    topk_acc.__name__ = 'topk_acc'
    # Compile model
    final_model.compile(loss='categorical_crossentropy',
                        metrics=[topk_acc],
                        optimizer=opt)

    # print model summary
    final_model.summary()

    # Get input shape
    image_height, image_width, nchannels = get_input_shape(final_model)
    print(image_height, image_width, nchannels)

    assert nchannels in [1, 3], (
        "Unsupported channel count {} for evaluation".format(nchannels)
    )
    color_mode = "rgb"
    if nchannels == 1:
        color_mode = "grayscale"
    interpolation = cfg['model_config']['resize_interpolation_method']
    if cfg['eval_config']['enable_center_crop']:
        interpolation += ":center"

    # Initializing data generator
    target_datagen = ImageDataGenerator(
        preprocessing_function=partial(preprocess_input,
                                       data_format=cfg['data_format'],
                                       mode=cfg['train_config']['preprocess_mode'],
                                       img_mean=list(cfg['train_config']['image_mean']),
                                       color_mode=color_mode),
        horizontal_flip=False,
        data_format=cfg['data_format'])
    # Initializing data iterator
    target_iterator = target_datagen.flow_from_directory(
        cfg['eval_config']['eval_dataset_path'],
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=cfg['eval_config']['batch_size'],
        class_mode='categorical',
        interpolation=interpolation,
        shuffle=False)
    # target_dataset = tf.data.Dataset.from_generator(target_iterator)
    logger.info('Processing dataset (evaluation): {}'.format(cfg['eval_config']['eval_dataset_path']))
    nclasses = target_iterator.num_classes
    assert nclasses > 1, "Invalid number of classes in the evaluation dataset."

    # If number of classes does not match the new data
    assert nclasses == final_model.output.get_shape().as_list()[-1], \
        "The number of classes of the loaded model doesn't match the \
         number of classes in the evaluation dataset."

    # Evaluate the model on the full data set.
    score = final_model.evaluate(target_iterator,
                                 steps=len(target_iterator),
                                 workers=cfg['eval_config']['n_workers'],
                                 use_multiprocessing=False)

    print('Evaluation Loss: {}'.format(score[0]))
    print('Evaluation Top K accuracy: {}'.format(score[1]))
    # Re-initializing data iterator
    target_iterator = target_datagen.flow_from_directory(
        cfg['eval_config']['eval_dataset_path'],
        target_size=(image_height, image_width),
        batch_size=cfg['eval_config']['batch_size'],
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


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application.
    """
    run_evaluate(cfg)
    logger.info("Training finished successfully.")


if __name__ == '__main__':
    main()

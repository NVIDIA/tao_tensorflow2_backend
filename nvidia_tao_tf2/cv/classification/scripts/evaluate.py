# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Perform classification evaluation."""
import os
from functools import partial
import logging
import json
import sys

import numpy as np
from PIL import ImageFile
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.logging.logging as status_logging
from nvidia_tao_tf2.common.decorators import monitor_status

from nvidia_tao_tf2.cv.classification.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.classification.utils import preprocess_crop  # noqa pylint: disable=unused-import
from nvidia_tao_tf2.cv.classification.utils.preprocess_input import preprocess_input
from nvidia_tao_tf2.cv.classification.utils.helper import get_input_shape, load_model
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='classification', mode='evaluation')
def run_evaluate(cfg):
    """Wrapper function to run evaluation of classification model.

    Args:
       Dictionary arguments containing parameters parsed in the main function.
    """
    # Set up logger verbosity.
    logger.setLevel(logging.INFO)
    # Decrypt EFF
    final_model = load_model(
        str(cfg.evaluate.model_path),
        cfg.encryption_key)

    # Defining optimizer
    opt = tf.keras.optimizers.legacy.SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=False)
    # Define precision/recall and F score metrics
    topk_acc = partial(tf.keras.metrics.top_k_categorical_accuracy,
                       k=cfg['evaluate']['top_k'])
    topk_acc.__name__ = 'topk_acc'
    # Compile model
    final_model.compile(loss='categorical_crossentropy',
                        metrics=[topk_acc],
                        optimizer=opt)

    # print model summary
    final_model.summary()

    # Get input shape
    image_height, image_width, nchannels = get_input_shape(final_model)
    image_depth = cfg['model']['input_image_depth']
    assert image_depth in [8, 16], "Only 8-bit and 16-bit images are supported"

    assert nchannels in [1, 3], (
        f"Unsupported channel count {nchannels} for evaluation"
    )
    logger.debug('input req HWC and depth: %s, %s, %s, %s', image_height, image_width, nchannels, image_depth)
    color_mode = "rgb"
    if nchannels == 1:
        color_mode = "grayscale"
    interpolation = cfg['model']['resize_interpolation_method']
    if cfg['augment']['enable_center_crop']:
        interpolation += ":center"

    # Initializing data generator
    target_datagen = ImageDataGenerator(
        preprocessing_function=partial(preprocess_input,
                                       data_format=cfg['data_format'],
                                       mode=cfg['data']['preprocess_mode'],
                                       img_mean=list(cfg['data']['image_mean']),
                                       color_mode=color_mode,
                                       img_depth=image_depth),
        horizontal_flip=False,
        data_format=cfg['data_format'])

    if cfg['evaluate']['classmap']:
        # If classmap is provided, then we explicitly set it in ImageDataGenerator
        with open(cfg['evaluate']['classmap'], "r", encoding='utf-8') as cmap_file:
            try:
                data = json.load(cmap_file)
            except json.decoder.JSONDecodeError as e:
                print(f"Loading the {cfg['evaluate']['classmap']} failed with error\n{e}")
                sys.exit(-1)
            except Exception as e:
                if e.output is not None:
                    print(f"Evaluation failed with error {e.output}")
                sys.exit(-1)
        if not data:
            class_names = None
            logger.info('classmap is not loaded.')
        else:
            class_names = [""] * len(list(data.keys()))
            if not all([class_index < len(class_names) and isinstance(class_index, int) # noqa pylint: disable=R1729
                        for class_index in data.values()]):
                raise RuntimeError(
                    "Invalid data in the json file. The class index must "
                    "be < number of classes and an integer value.")
            for class_name, class_index in data.items():
                class_names[class_index] = class_name
            logger.info('classmap is loaded successfully.')
    else:
        class_names = None

    # Initializing data iterator
    target_iterator = target_datagen.flow_from_directory(
        cfg['evaluate']['dataset_path'],
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=cfg['evaluate']['batch_size'],
        class_mode='categorical',
        interpolation=interpolation,
        shuffle=False)

    logger.info('Processing dataset (evaluation): {}'.format(cfg['evaluate']['dataset_path']))  # noqa pylint: disable=C0209
    nclasses = target_iterator.num_classes
    assert nclasses > 1, "Invalid number of classes in the evaluation dataset."

    # If number of classes does not match the new data
    assert nclasses == final_model.output.get_shape().as_list()[-1], \
        "The number of classes of the loaded model doesn't match the \
         number of classes in the evaluation dataset."

    # Evaluate the model on the full data set.
    score = final_model.evaluate(target_iterator,
                                 steps=len(target_iterator),
                                 workers=cfg['evaluate']['n_workers'],
                                 use_multiprocessing=False)

    logger.info('Evaluation Loss: %s', score[0])
    logger.info('Evaluation Top %s accuracy: %s', cfg['evaluate']['top_k'], score[1])
    status_logging.get_status_logger().kpi['loss'] = float(score[0])
    status_logging.get_status_logger().kpi['top_k'] = float(score[1])
    # Re-initializing data iterator
    target_iterator = target_datagen.flow_from_directory(
        cfg['evaluate']['dataset_path'],
        target_size=(image_height, image_width),
        batch_size=cfg['evaluate']['batch_size'],
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
    target_keys_names = list(sorted(class_dict.items(), key=lambda x: x[1]))
    target_keys_names = list(zip(*target_keys_names))
    print(classification_report(target_iterator.classes, y_pred, labels=target_keys_names[1], target_names=target_keys_names[0]))


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application."""
    run_evaluate(cfg)


if __name__ == '__main__':
    main()

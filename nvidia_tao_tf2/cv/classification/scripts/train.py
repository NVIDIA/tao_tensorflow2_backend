# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Classification training script with protobuf configuration."""
from functools import partial
import json
import logging
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile

import horovod.tensorflow.keras as hvd

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_tf2.common.mlops.utils import init_mlops
from nvidia_tao_tf2.common.utils import set_random_seed, update_results_dir

from nvidia_tao_tf2.cv.classification.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.classification.model.classifier_module import ClassifierModule
from nvidia_tao_tf2.cv.classification.model.callback_builder import setup_callbacks
from nvidia_tao_tf2.cv.classification.trainer.classifier_trainer import ClassifierTrainer

from nvidia_tao_tf2.cv.classification.utils.config_utils import spec_checker
from nvidia_tao_tf2.cv.classification.utils.mixup_generator import MixupImageDataGenerator
from nvidia_tao_tf2.cv.classification.utils.preprocess_input import preprocess_input
from nvidia_tao_tf2.cv.classification.utils import preprocess_crop  # noqa pylint: disable=unused-import
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 9000000000
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level='INFO')
logger = logging.getLogger(__name__)


def setup_env(cfg):
    """Setup training env."""
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Configure the logger.
    logger.setLevel(logging.INFO)
    # Set random seed.
    seed = cfg.train.random_seed + hvd.rank()
    set_random_seed(seed)
    logger.debug("Random seed is set to %d", seed)
    # Create results dir
    if hvd.rank() == 0:
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir, exist_ok=True)
        init_mlops(cfg, name='classification')


def load_data(train_data,
              val_data,
              cfg,
              batch_size=8,
              enable_random_crop=False,
              enable_center_crop=False,
              enable_color_augmentation=False,
              interpolation='bicubic',
              num_classes=1000,
              mixup_alpha=0.0,
              no_horizontal_flip=False,
              data_format='channels_first'):
    """Load training and validation data with default data augmentation.

    Args:
        train_data (str): path to the training data.
        val_data (str): path to the validation data.
        preprocessing_func: function to process an image.
        batch_size (int): Number of image tensors per batch.
        enable_random_crop (bool): Flag to enable random cropping in load_img.
        enable_center_crop (bool): Flag to enable center cropping for val.
        interpolation(str): Interpolation method for image resize. choices: `bilinear` or `bicubic`.
        num_classes (int): Number of classes.
        mixup_alpha (float): mixup alpha.
        no_horizontal_flip(bool): Flag to disable horizontal flip for
            direction-aware datasets.
    Return:
        train/val Iterators and number of classes in the dataset.
    """
    image_depth = cfg.model.input_image_depth
    color_mode = "rgb" if cfg.model.input_channels == 3 else "grayscale"

    preprocessing_func = partial(
        preprocess_input,
        data_format=cfg.data_format,
        mode=cfg.dataset.preprocess_mode,
        img_mean=list(cfg.dataset.image_mean),
        color_mode=color_mode,
        img_depth=image_depth)

    preprocess_crop._set_color_augmentation(enable_color_augmentation)
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
        cfg.model.input_height, cfg.model.input_width,
        color_mode=color_mode,
        interpolation=interpolation + ':random' if enable_random_crop else interpolation,
        alpha=mixup_alpha
    )
    logger.info('Processing dataset (train): %s', train_data)

    # Initializing data generator: Val
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=False,
        data_format=data_format)

    # Initializing data iterator: Val
    val_iterator = val_datagen.flow_from_directory(
        val_data,
        target_size=(cfg.model.input_height, cfg.model.input_width),
        color_mode=color_mode,
        batch_size=batch_size,
        interpolation=interpolation + ':center' if enable_center_crop else interpolation,
        shuffle=False,
        class_mode='categorical')
    logger.info('Processing dataset (validation): %s', val_data)

    # Check if the number of classes is consistent
    assert train_iterator.num_classes == val_iterator.num_classes == num_classes, \
        "Number of classes in train and val don't match."
    return train_iterator, val_iterator


@monitor_status(name='classification', mode='training')
def run_experiment(cfg):
    """Launch training experiment."""
    spec_checker(cfg)
    # Load augmented data
    train_iterator, val_iterator = load_data(
        cfg.dataset.train_dataset_path,
        cfg.dataset.val_dataset_path,
        cfg=cfg,
        batch_size=cfg.train.batch_size_per_gpu,
        enable_random_crop=cfg.dataset.augmentation.enable_random_crop,
        enable_center_crop=cfg.dataset.augmentation.enable_center_crop,
        enable_color_augmentation=cfg.dataset.augmentation.enable_color_augmentation,
        interpolation=cfg.model.resize_interpolation_method,
        num_classes=cfg.model.num_classes,
        mixup_alpha=cfg.dataset.augmentation.mixup_alpha,
        no_horizontal_flip=cfg.dataset.augmentation.disable_horizontal_flip,
        data_format=cfg.data_format)

    # Initialize classifier module
    steps_per_epoch = (len(train_iterator) + hvd.size() - 1) // hvd.size()
    classifier = ClassifierModule(cfg, steps_per_epoch)
    # Setup callbacks
    callbacks = setup_callbacks(
        cfg.train.checkpoint_interval,
        cfg.results_dir,
        cfg.train.lr_config,
        classifier.initial_epoch + 1,
        classifier.steps_per_epoch,
        cfg.train.num_epochs,
        cfg.encryption_key)
    # Writing out class-map file for inference mapping
    if hvd.rank() == 0:
        with open(os.path.join(cfg.results_dir, "classmap.json"), "w", encoding='utf-8') as f:
            json.dump(train_iterator.class_indices, f)
        logger.info('classmap.json is generated at %s', cfg.results_dir)
    # Initialize classifier trainer
    trainer = ClassifierTrainer(
        num_epochs=cfg.train.num_epochs,
        callbacks=callbacks,
        cfg=cfg)
    trainer.fit(
        module=classifier,
        train_dataset=train_iterator,
        eval_dataset=val_iterator,
        verbose=1 if hvd.rank() == 0 else 0
    )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application."""
    cfg = update_results_dir(cfg, 'train')
    setup_env(cfg)
    run_experiment(cfg=cfg)


if __name__ == '__main__':
    main()

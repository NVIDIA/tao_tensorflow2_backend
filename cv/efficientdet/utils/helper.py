# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Collection of helper functions."""
import os

import tensorflow as tf
import numpy as np
import tempfile
import zipfile

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback

from cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion
from cv.efficientdet.utils import keras_utils

CUSTOM_OBJS = {
    'ImageResizeLayer': ImageResizeLayer,
    'WeightedFusion': WeightedFusion}


def fetch_optimizer(model,opt_type) -> tf.keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""
    
    # this is the case where our target optimizer is not wrapped by any other optimizer(s)
    if isinstance(model.optimizer,opt_type):
        return model.optimizer
    
    # Dive into nested optimizer object until we reach the target opt
    opt = model.optimizer
    while hasattr(opt, '_optimizer'):
        opt = opt._optimizer
        if isinstance(opt,opt_type):
            return opt 
    raise TypeError(f'Failed to find {opt_type} in the nested optimizer object')


def decode_eff(eff_model_path, passphrase=None):
    """Decode EFF to saved_model directory.

    Args:
        eff_model_path (str): Path to eff model
        passphrase (str, optional): Encryption key. Defaults to None.

    Returns:
        str: Path to the saved_model
    """
    # Decrypt EFF
    eff_filename = os.path.basename(eff_model_path)
    eff_art = Archive.restore_artifact(
        restore_path=eff_model_path,
        artifact_name=eff_filename,
        passphrase=passphrase)
    zip_path = eff_art.get_handle()
    # Unzip
    saved_model_path = os.path.dirname(zip_path)
    # TODO(@yuw): try catch? 
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(saved_model_path)
    return saved_model_path


def load_model(model_path, cfg):
    """Load hdf5 or EFF model.

    Args:
        model_path (str): Path to EfficientDet checkpoint
        passphrase (str, optional): Encryption key. Defaults to None.

    Returns:
        Keras model: Loaded model
    """
    is_pruned = False
    if is_pruned:
        raise NotImplementedError
    else:
        # model_path is saved_model
        model = load_json_model(os.path.join(cfg['results_dir'], 'model_graph.json'))
        train_from_epoch = keras_utils.restore_ckpt(
            model,
            model_path, 
            cfg.train_config.moving_average_decay,
            steps_per_epoch=0,
            expect_partial=False)
        # TODO(@yuw): verify train_from_epoch
        return model
            


def load_json_model(json_path, new_objs=None):
    """Helper function to load keras model from json file."""
    new_objs = new_objs or {}
    with open(json_path, 'r') as jf:
        model_json = jf.read()
    loaded_model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={**CUSTOM_OBJS, **new_objs})
    return loaded_model


def dump_json(model, out_path):
    """Model to json."""
    with open(out_path, "w") as jf:
        jf.write(model.to_json())


def zipdir(src, zip_path):
    """Function creates zip archive from src in dst location.
    
    Args:
        src: Path to directory to be archived.
        dst: Path where archived dir will be stored.
    """
    # destination directory
    os.chdir(os.path.dirname(zip_path))
    # zipfile handler
    with zipfile.ZipFile(zip_path, "w") as zf:
        ### writing content of src directory to the archive
        for root, _, filenames in os.walk(src):
            for filename in filenames:
                zf.write(
                    os.path.join(root, filename),
                    arcname=os.path.join(root.replace(src, ""),
                    filename))


def encode_eff(filepath, eff_model_path, passphrase):
    """Encode saved_model directory into a .eff file.

    Args:
        filepath (str): Path to saved_model
        eff_model_path (str): Path to the output EFF file
        passphrase (str): Encrytion key
    """
    os_handle, temp_zip_file = tempfile.mkstemp()
    os.close(os_handle)
    # create zipfile from saved_model directory
    zipdir(filepath, temp_zip_file)
    # create artifacts from zipfile
    eff_filename = os.path.basename(eff_model_path)
    zip_art = File(
        name=eff_filename,
        description="Artifact from checkpoint",
        filepath=temp_zip_file,
        content_callback=BinaryContentCallback,
    )
    Archive.save_artifact(
        save_path=eff_model_path, artifact=zip_art, passphrase=passphrase)
    return temp_zip_file
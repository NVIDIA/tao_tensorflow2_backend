# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Collection of helper functions."""
import os

import tensorflow import keras
import tensorflow as tf
import numpy as np
import tempfile
import zipfile

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback


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


def load_model(model_path, passphrase=None):
    """Load hdf5 or EFF model.

    Args:
        model_path (str): Path to hdf5 model or eff model
        passphrase (str, optional): Encryption key. Defaults to None.

    Returns:
        Keras model: Loaded model
    """
    assert os.path.exists(model_path), "Pretrained model not found at {}".format(model_path)
    assert os.path.splitext(model_path)[-1] in ['.hdf5', '.eff'], \
        "Only .hdf5 and .tlt are supported."
    if model_path.endswith('.eff'):
        model_path = decode_eff(model_path, passphrase)
    return tf.keras.models.load_model(model_path)


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

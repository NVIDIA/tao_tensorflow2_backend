# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Collection of helper functions."""
import os
import json
import tensorflow as tf
import tempfile
import zipfile

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback

from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase
from tensorflow_quantization.quantize import quantize_model

from nvidia_tao_tf2.cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from nvidia_tao_tf2.cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion
from nvidia_tao_tf2.cv.efficientdet.utils import keras_utils

CUSTOM_OBJS = {
    'ImageResizeLayer': ImageResizeLayer,
    'WeightedFusion': WeightedFusion,
}


def fetch_optimizer(model, opt_type) -> tf.keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""
    # this is the case where our target optimizer is not wrapped by any other optimizer(s)
    if isinstance(model.optimizer, opt_type):
        return model.optimizer

    # Dive into nested optimizer object until we reach the target opt
    opt = model.optimizer
    while hasattr(opt, '_optimizer'):
        opt = opt._optimizer
        if isinstance(opt, opt_type):
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
    ckpt_path = os.path.dirname(zip_path)
    # TODO(@yuw): try catch?
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(ckpt_path)
    extracted_files = os.listdir(ckpt_path)
    ckpt_name = None
    for f in extracted_files:
        if 'ckpt' in f:
            ckpt_name = f.split('.')[0]
    # TODO(@yuw): backbone ckpt vs effdet vs failed case
    # if not ckpt_name:
    #     raise IOError(f"{eff_model_path} was not saved properly.")
    return ckpt_path, ckpt_name


def load_model(eff_model_path, hparams, mode='train', is_qat=False):
    """Load hdf5 or EFF model.

    Args:
        model_path (str): Path to EfficientDet checkpoint
        passphrase (str, optional): Encryption key. Defaults to None.

    Returns:
        Keras model: Loaded model
    """
    ckpt_path, ckpt_name = decode_eff(eff_model_path, hparams.encryption_key)
    if mode != 'train':
        mode = 'eval'
    model = load_json_model(
        os.path.join(ckpt_path, f'{mode}_graph.json'))
    if is_qat:
        model = quantize_model(model, custom_qdq_cases=[EfficientNetQDQCase()])
    keras_utils.restore_ckpt(
        model,
        os.path.join(ckpt_path, ckpt_name),
        hparams.train.moving_average_decay,
        steps_per_epoch=0,
        expect_partial=True)
    return model


def load_json_model(json_path, new_objs=None):
    """Helper function to load keras model from json file."""
    new_objs = new_objs or {}
    with open(json_path, 'r', encoding='utf-8') as jf:
        model_json = jf.read()
    loaded_model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={**CUSTOM_OBJS, **new_objs})
    return loaded_model


def dump_json(model, out_path):
    """Model to json."""
    json_str = model.to_json()
    model_json = json.loads(json_str)
    # workaround to remove float16 dtype if trained with AMP
    for layer in model_json['config']['layers']:
        if isinstance(layer['config']['dtype'], dict):
            layer['config']['dtype'] = 'float32'
    with open(out_path, "w", encoding='utf-8') as jf:
        json.dump(model_json, jf)


def dump_eval_json(graph_dir, train_graph="train_graph.json", eval_graph='eval_graph.json'):
    """Generate and save the evaluation graph by modifying train graph.

    Args:
        graph_dir (str): Directory where the train graph resides in.
    """
    # generate eval graph for exporting. (time saving hack)
    with open(os.path.join(graph_dir, train_graph), 'r', encoding='utf-8') as f:
        pruned_json = json.load(f)
        for layer in pruned_json['config']['layers']:
            if layer['class_name'] == 'BatchNormalization':
                if layer['inbound_nodes'][0][0][-1]:
                    layer['inbound_nodes'][0][0][-1]['training'] = False
    with open(os.path.join(graph_dir, eval_graph), 'w', encoding='utf-8') as jf:
        json.dump(pruned_json, jf)


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
        # writing content of src directory to the archive
        for root, _, filenames in os.walk(src):
            for filename in filenames:
                zf.write(
                    os.path.join(root, filename),
                    arcname=os.path.join(root.replace(src, ""), filename))


def encode_eff(filepath, eff_model_path, passphrase, is_pruned=False):
    """Encode saved_model directory into a .tlt file.

    Args:
        filepath (str): Path to saved_model
        eff_model_path (str): Path to the output EFF file
        passphrase (str): Encryption key
    """
    # always overwrite
    if os.path.exists(eff_model_path):
        os.remove(eff_model_path)
    os_handle, temp_zip_file = tempfile.mkstemp()
    os.close(os_handle)
    # create zipfile from saved_model directory
    zipdir(filepath, temp_zip_file)
    # create artifacts from zipfile
    eff_filename = os.path.basename(eff_model_path)
    zip_art = File(
        name=eff_filename,
        is_pruned=is_pruned,
        description="Artifact from checkpoint",
        filepath=temp_zip_file,
        encryption=True,
        content_callback=BinaryContentCallback,
    )
    Archive.save_artifact(
        save_path=eff_model_path, artifact=zip_art, passphrase=passphrase)
    return temp_zip_file

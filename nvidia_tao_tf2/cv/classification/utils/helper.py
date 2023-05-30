# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Collection of helper functions."""
import os

import cv2
import importlib
import json
import warnings

from tensorflow import keras
import tensorflow as tf
from numba import errors
from numba import jit, njit
import numpy as np

from PIL import Image
import tempfile
import zipfile

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback

from nvidia_tao_tf2.common.utils import (
    CUSTOM_OBJS,
    MultiGPULearningRateScheduler,
    SoftStartCosineAnnealingScheduler,
    StepLRScheduler
)

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.simplefilter('ignore', category=errors.NumbaWarning)
warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)

opt_dict = {
    'sgd': keras.optimizers.legacy.SGD,
    'adam': keras.optimizers.legacy.Adam,
    'rmsprop': keras.optimizers.legacy.RMSprop
}

scope_dict = {'dense': keras.layers.Dense,
              'conv2d': keras.layers.Conv2D}

regularizer_dict = {'l1': keras.regularizers.l1,
                    'l2': keras.regularizers.l2}


def initialize():
    """Initializes backend related initializations."""
    if tf.config.list_physical_devices('GPU'):
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)


def build_optimizer(optimizer_config):
    """build optimizer with the optimizer config."""
    if optimizer_config.optimizer == "sgd":
        return opt_dict["sgd"](
            learning_rate=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            decay=optimizer_config.decay,
            nesterov=optimizer_config.nesterov
        )
    if optimizer_config.optimizer == "adam":
        return opt_dict["adam"](
            learning_rate=optimizer_config.lr,
            beta_1=optimizer_config.beta_1,
            beta_2=optimizer_config.beta_2,
            epsilon=optimizer_config.epsilon,
            decay=optimizer_config.decay
        )
    if optimizer_config.optimizer == "rmsprop":
        return opt_dict["rmsprop"](
            learning_rate=optimizer_config.lr,
            rho=optimizer_config.rho,
            epsilon=optimizer_config.epsilon,
            decay=optimizer_config.decay
        )
    raise ValueError(f"Unsupported Optimizer: {optimizer_config.optimizer}")


def build_lr_scheduler(lr_config, hvd_size, max_iterations):
    """Build a learning rate scheduler from config."""
    # Set up the learning rate callback. It will modulate learning rate
    # based on iteration progress to reach max_iterations.
    if lr_config.scheduler == 'step':
        lrscheduler = StepLRScheduler(
            base_lr=lr_config.learning_rate * hvd_size,
            gamma=lr_config.gamma,
            step_size=lr_config.step_size,
            max_iterations=max_iterations
        )
    elif lr_config.scheduler == 'soft_anneal':
        lrscheduler = MultiGPULearningRateScheduler(
            base_lr=lr_config.learning_rate * hvd_size,
            soft_start=lr_config.soft_start,
            annealing_points=lr_config.annealing_points,
            annealing_divider=lr_config.annealing_divider,
            max_iterations=max_iterations
        )
    elif lr_config.scheduler == 'cosine':
        lrscheduler = SoftStartCosineAnnealingScheduler(
            base_lr=lr_config.learning_rate * hvd_size,
            min_lr_ratio=lr_config.min_lr_ratio,
            soft_start=lr_config.soft_start,
            max_iterations=max_iterations
        )
    else:
        raise ValueError(
            f"Only `step`, `cosine` and `soft_anneal`. LR scheduler are supported, but {lr_config.scheduler} is specified."
        )
    return lrscheduler


def get_input_shape(model, data_format):
    """Obtain input shape from a Keras model."""
    # Computing shape of input tensor
    image_shape = model.layers[0].input_shape[0][1:4]
    # Setting input shape
    if data_format == "channels_first":
        nchannels, image_height, image_width = image_shape[0:3]
    else:
        image_height, image_width, nchannels = image_shape[0:3]
    return image_height, image_width, nchannels


@njit
def randu(low, high):
    """standard uniform distribution."""
    return np.random.random() * (high - low) + low


@jit
def random_hue(img, max_delta=10.0):
    """Rotates the hue channel.

    Args:
        img: input image in float32
        max_delta: Max number of degrees to rotate the hue channel
    """
    # Rotates the hue channel by delta degrees
    delta = randu(-max_delta, max_delta)
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hchannel = hsv[:, :, 0]
    hchannel = delta + hchannel
    # hue should always be within [0,360]
    idx = np.where(hchannel > 360)
    hchannel[idx] = hchannel[idx] - 360
    idx = np.where(hchannel < 0)
    hchannel[idx] = hchannel[idx] + 360
    hsv[:, :, 0] = hchannel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@jit
def random_saturation(img, max_shift):
    """random saturation data augmentation."""
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    shift = randu(-max_shift, max_shift)
    # saturation should always be within [0,1.0]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + shift, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@jit
def random_contrast(img, center, max_contrast_scale):
    """random contrast data augmentation."""
    new_img = (img - center) * (1.0 + randu(-max_contrast_scale, max_contrast_scale)) + center
    new_img = np.clip(new_img, 0., 1.)
    return new_img


@jit
def random_shift(x_img, shift_stddev):
    """random shift data augmentation."""
    shift = np.random.randn() * shift_stddev
    new_img = np.clip(x_img + shift, 0.0, 1.0)

    return new_img


def color_augmentation(
    x_img,
    color_shift_stddev=0.0,
    hue_rotation_max=25.0,
    saturation_shift_max=0.2,
    contrast_center=0.5,
    contrast_scale_max=0.1
):
    """color augmentation for images."""
    # convert PIL Image to numpy array
    x_img = np.array(x_img, dtype=np.float32)
    # normalize the image to (0, 1)
    x_img /= 255.0
    x_img = random_shift(x_img, color_shift_stddev)
    x_img = random_hue(x_img, max_delta=hue_rotation_max)
    x_img = random_saturation(x_img, saturation_shift_max)
    x_img = random_contrast(
        x_img,
        contrast_center,
        contrast_scale_max
    )
    # convert back to PIL Image
    x_img *= 255.0
    return Image.fromarray(x_img.astype(np.uint8), "RGB")


def setup_config(model, reg_config, bn_config=None, custom_objs=None):
    """Wrapper for setting up BN and regularizer.

    Args:
        model (keras Model): a Keras model
        reg_config (dict): reg_config dict
        bn_config (dict): config to override BatchNormalization parameters
        custom_objs (dict): Custom objects for serialization and deserialization.
    Return:
        A new model with overridden config.
    """
    if bn_config is not None:
        bn_momentum = bn_config['momentum']
        bn_epsilon = bn_config['epsilon']
    else:
        bn_momentum = 0.9
        bn_epsilon = 1e-5
    # Obtain the current configuration from model
    mconfig = model.get_config()
    # Obtain type and scope of the regularizer
    reg_type = reg_config['type'].lower()
    scope_list = reg_config['scope']
    layer_list = [scope_dict[i.lower()] for i in scope_list if i.lower()
                  in scope_dict]

    for layer, layer_config in zip(model.layers, mconfig['layers']):
        # BN settings
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = bn_momentum
            layer_config['config']['epsilon'] = bn_epsilon

        # Regularizer settings
        if reg_type:
            if type(layer) in layer_list and \
               hasattr(layer, 'kernel_regularizer'):

                assert reg_type in ['l1', 'l2', 'none'], \
                    "Regularizer can only be either L1, L2 or None."

                if reg_type in ['l1', 'l2']:
                    assert 0 < reg_config['weight_decay'] < 1, \
                        "Weight decay should be no less than 0 and less than 1"
                    regularizer = regularizer_dict[reg_type](
                        reg_config['weight_decay'])
                    layer_config['config']['kernel_regularizer'] = \
                        {'class_name': regularizer.__class__.__name__,
                         'config': regularizer.get_config()}

                if reg_type == 'none':
                    layer_config['config']['kernel_regularizer'] = None

    if custom_objs:
        CUSTOM_OBJS.update(custom_objs)

    with keras.utils.CustomObjectScope(CUSTOM_OBJS):
        updated_model = keras.models.Model.from_config(mconfig)
    updated_model.set_weights(model.get_weights())

    return updated_model


def decode_eff(eff_model_path, enc_key=None):
    """Decode EFF to saved_model directory.

    Args:
        eff_model_path (str): Path to eff model
        enc_key (str, optional): Encryption key. Defaults to None.

    Returns:
        str: Path to the saved_model
    """
    # Decrypt EFF
    eff_filename = os.path.basename(eff_model_path)
    eff_art = Archive.restore_artifact(
        restore_path=eff_model_path,
        artifact_name=eff_filename,
        passphrase=enc_key)
    zip_path = eff_art.get_handle()
    # Unzip
    saved_model_path = os.path.dirname(zip_path)
    # TODO(@yuw): try catch
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(saved_model_path)
    return saved_model_path


def deserialize_custom_layers(art):
    """Deserialize the code for custom layer from EFF.

    Args:
        art (eff.core.artifact.Artifact): Artifact restored from EFF Archive.

    Returns:
        final_dict (dict): Dictionary representing CUSTOM_OBJS used in the EFF stored Keras model.
    """
    # Get class.
    source_code = art.get_content()
    spec = importlib.util.spec_from_loader('helper', loader=None)
    helper = importlib.util.module_from_spec(spec)
    exec(source_code, helper.__dict__) # noqa pylint: disable=W0122

    final_dict = {}
    # Get class name from attributes.
    class_names = art["class_names"]
    for cn in class_names:
        final_dict[cn] = getattr(helper, cn)
    return final_dict


def decode_tltb(eff_path, enc_key=None):
    """Restore Keras Model from EFF Archive.

    Args:
        eff_path (str): Path to the eff file.
        enc_key (str): Key to load EFF file.

    Returns:
        model (keras.models.Model): Loaded keras model.
        EFF_CUSTOM_OBJS (dict): Dictionary of custom layers from the eff file.
    """
    model_name = os.path.basename(eff_path).split(".")[0]
    with Archive.restore_from(restore_path=eff_path, passphrase=enc_key) as restored_effa:
        EFF_CUSTOM_OBJS = deserialize_custom_layers(restored_effa.artifacts['custom_layers.py'])
        model_name = restored_effa.metadata['model_name']

        art = restored_effa.artifacts[f'{model_name}.hdf5']
        weights, m = art.get_content()

    m = json.loads(m)
    with keras.utils.CustomObjectScope(EFF_CUSTOM_OBJS):
        model = keras.models.model_from_config(m, custom_objects=EFF_CUSTOM_OBJS)
        model.set_weights(weights)

    result = {
        "model": model,
        "custom_objs": EFF_CUSTOM_OBJS,
        "model_name": model_name
    }
    return result


def load_model(model_path, enc_key=None):
    """Load hdf5 or EFF model.

    Args:
        model_path (str): Path to hdf5 model or eff model
        enc_key (str, optional): Encryption key. Defaults to None.

    Returns:
        Keras model: Loaded model
    """
    assert os.path.exists(model_path), f"Pretrained model not found at {model_path}"
    if model_path.endswith('.tlt'):
        model_path = decode_eff(model_path, enc_key)
        return tf.keras.models.load_model(model_path)
    if model_path.endswith('.tltb'):
        out_dict = decode_tltb(model_path, enc_key)
        model = out_dict['model']
        return model
    return tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJS)


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


def encode_eff(filepath, eff_model_path, enc_key):
    """Encode saved_model directory into a .tlt file.

    Args:
        filepath (str): Path to saved_model
        eff_model_path (str): Path to the output EFF file
        enc_key (str): Encrytion key
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
        encryption=bool(enc_key),
        content_callback=BinaryContentCallback,
    )
    Archive.save_artifact(
        save_path=eff_model_path, artifact=zip_art, passphrase=enc_key)

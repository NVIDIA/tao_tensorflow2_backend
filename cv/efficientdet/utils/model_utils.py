"""Model utils."""
import contextlib
import logging
from typing import Text, Tuple, Union
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
# pylint: disable=logging-format-interpolation


def get_ema_vars():
    """Get all exponential moving average (ema) variables."""
    ema_vars = tf.trainable_variables() + tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
    for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            ema_vars.append(v)
    return list(set(ema_vars))


def get_ckpt_var_map(ckpt_path, ckpt_scope, var_scope, skip_mismatch=None):
    """Get a var map for restoring from pretrained checkpoints.

    Args:
        ckpt_path: string. A pretrained checkpoint path.
        ckpt_scope: string. Scope name for checkpoint variables.
        var_scope: string. Scope name for model variables.
        skip_mismatch: skip variables if shape mismatch.

    Returns:
        var_map: a dictionary from checkpoint name to model variables.
    """
    logging.info('Init model from checkpoint %s', ckpt_path)
    if not ckpt_scope.endswith('/') or not var_scope.endswith('/'):
        raise ValueError('Please specific scope name ending with /')
    if ckpt_scope.startswith('/'):
        ckpt_scope = ckpt_scope[1:]
    if var_scope.startswith('/'):
        var_scope = var_scope[1:]

    var_map = {}
    # Get the list of vars to restore.
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
    reader = tf.train.load_checkpoint(ckpt_path)
    ckpt_var_name_to_shape = reader.get_variable_to_shape_map()
    ckpt_var_names = set(reader.get_variable_to_shape_map().keys())

    for i, v in enumerate(model_vars):
        if not v.op.name.startswith(var_scope):
            logging.info('skip %s -- does not match scope %s', v.op.name, var_scope)
        ckpt_var = ckpt_scope + v.op.name[len(var_scope):]
        if (ckpt_var not in ckpt_var_names and
                v.op.name.endswith('/ExponentialMovingAverage')):
            ckpt_var = ckpt_scope + v.op.name[:-len('/ExponentialMovingAverage')]

        if ckpt_var not in ckpt_var_names:
            if 'Momentum' in ckpt_var or 'RMSProp' in ckpt_var:
                # Skip optimizer variables.
                continue
            if skip_mismatch:
                logging.info('skip %s (%s) -- not in ckpt', v.op.name, ckpt_var)
                continue
            raise ValueError(f'{v.op} is not in ckpt {ckpt_path}')

        if v.shape != ckpt_var_name_to_shape[ckpt_var]:
            if skip_mismatch:
                logging.info('skip %s (%s vs %s) -- shape mismatch', v.op.name, v.shape, ckpt_var_name_to_shape[ckpt_var])
                continue
            raise ValueError(f'shape mismatch {v.op.name} ({v.shape} vs {ckpt_var_name_to_shape[ckpt_var]})')

        if i < 5:
            # Log the first few elements for sanity check.
            logging.info('Init %s from ckpt var %s', v.op.name, ckpt_var)
        var_map[ckpt_var] = v

    return var_map


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = inputs / survival_prob * binary_tensor
    return output


def num_params_flops(readable_format=True):
    """Return number of parameters and flops."""
    nparams = np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    options = tf.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    flops = tf.profiler.profile(
        tf.get_default_graph(), options=options).total_float_ops
    # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
    flops = flops // 2
    if readable_format:
        nparams = float(nparams) * 1e-6
        flops = float(flops) * 1e-9
    return nparams, flops


def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
    """Parse the image size and return (height, width).

    Args:
        image_size: A integer, a tuple (H, W), or a string with HxW format.

    Returns:
        A tuple of integer (height, width).
    """
    if isinstance(image_size, int):
        # image_size is integer, with the same width and height.
        return (image_size, image_size)

    if isinstance(image_size, str):
        # image_size is a string with format WxH
        width, height = image_size.lower().split('x')
        return (int(height), int(width))

    if isinstance(image_size, tuple):
        return image_size

    raise ValueError(f'image_size must be an int, WxH string, or (height, width) tuple. Was {image_size}')


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]],
                   max_level: int):
    """Get feat widths and heights for all levels.

    Args:
        image_size: A integer, a tuple (H, W), or a string with HxW format.
        max_level: maximum feature level.

    Returns:
        feat_sizes: a list of tuples (height, width) for each level.
    """
    image_size = parse_image_size(image_size)
    feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
    feat_size = image_size
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
    return feat_sizes


def verify_feats_size(feats,
                      feat_sizes,
                      min_level,
                      max_level,
                      data_format='channels_last'):
    """Verify the feature map sizes."""
    expected_output_size = feat_sizes[min_level:max_level + 1]
    for cnt, size in enumerate(expected_output_size):
        h_id, w_id = (2, 3) if data_format == 'channels_first' else (1, 2)
        if feats[cnt].shape[h_id] != size['height']:
            raise ValueError(
                f"feats[{cnt}] has shape {feats[cnt].shape} but its height should be {size['height']}. "
                "(input_height: {feat_sizes[0]['height']}, min_level: {min_level}, max_level: {max_level}.)")
        if feats[cnt].shape[w_id] != size['width']:
            raise ValueError(
                f"feats[{cnt}] has shape {feats[cnt].shape} but its width should be {size['width']}."
                "(input_width: {feat_sizes[0]['width']}, min_level: {min_level}, max_level: {max_level}.)")


@contextlib.contextmanager
def float16_scope():
    """Scope class for float16."""

    def _custom_getter(getter, *args, **kwargs):
        """Returns a custom getter that methods must be called under."""
        cast_to_float16 = False
        requested_dtype = kwargs['dtype']
        if requested_dtype == tf.float16:
            kwargs['dtype'] = tf.float32
            cast_to_float16 = True
        var = getter(*args, **kwargs)
        if cast_to_float16:
            var = tf.cast(var, tf.float16)
        return var

    with tf.variable_scope('', custom_getter=_custom_getter) as varscope:
        yield varscope

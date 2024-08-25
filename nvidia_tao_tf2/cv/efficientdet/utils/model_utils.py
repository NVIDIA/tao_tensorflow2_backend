"""Model utils."""
import contextlib
from typing import Text, Tuple, Union
import numpy as np
import tensorflow as tf
# pylint: disable=logging-format-interpolation


def num_params(model):
    """Return number of parameters."""
    return np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


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

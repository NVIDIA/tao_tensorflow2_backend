"""Model utils."""
import contextlib
from typing import Text, Tuple, Union
from absl import logging
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
    logging.info(f'Init model from checkpoint {ckpt_path}')
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
            logging.info(f'skip {v.op.name} -- does not match scope {var_scope}')
        ckpt_var = ckpt_scope + v.op.name[len(var_scope):]
        if (ckpt_var not in ckpt_var_names and
                v.op.name.endswith('/ExponentialMovingAverage')):
            ckpt_var = ckpt_scope + v.op.name[:-len('/ExponentialMovingAverage')]

        if ckpt_var not in ckpt_var_names:
            if 'Momentum' in ckpt_var or 'RMSProp' in ckpt_var:
                # Skip optimizer variables.
                continue
            if skip_mismatch:
                logging.info(f'skip {v.op.name} ({ckpt_var}) -- not in ckpt')
                continue
            raise ValueError(f'{v.op} is not in ckpt {ckpt_path}')

        if v.shape != ckpt_var_name_to_shape[ckpt_var]:
            if skip_mismatch:
                logging.info(f'skip {v.op.name} ({v.shape} vs {ckpt_var_name_to_shape[ckpt_var]}) -- shape mismatch')
                continue
            raise ValueError(f'shape mismatch {v.op.name} ({v.shape} vs {ckpt_var_name_to_shape[ckpt_var]})')

        if i < 5:
            # Log the first few elements for sanity check.
            logging.info(f'Init {v.op.name} from ckpt var {ckpt_var}')
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


conv_kernel_initializer = tf.initializers.variance_scaling()
dense_kernel_initializer = tf.initializers.variance_scaling()


class Pair(tuple):
    """Custom Pair class."""

    def __new__(cls, name, value):
        """New."""
        return super().__new__(cls, (name, value))

    def __init__(self, name, _):  # pylint: disable=super-init-not-called
        """Init."""
        self.name = name


def scalar(name, tensor, is_tpu=True):
    """Stores a (name, Tensor) tuple in a custom collection."""
    logging.info(f'Adding scale summary {Pair(name, tensor)}')
    if is_tpu:
        tf.add_to_collection('scalar_summaries', Pair(name, tf.reduce_mean(tensor)))
    else:
        tf.summary.scalar(name, tf.reduce_mean(tensor))


def image(name, tensor, is_tpu=True):
    """Store a (name, Tensor) tuple in a custom collection."""
    logging.info(f'Adding image summary {Pair(name, tensor)}')
    if is_tpu:
        tf.add_to_collection('image_summaries', Pair(name, tensor))
    else:
        tf.summary.image(name, tensor)


def get_tpu_host_call(global_step, params):
    """Get TPU host call for summaries."""
    scalar_summaries = tf.get_collection('scalar_summaries')
    if params['img_summary_steps']:
        image_summaries = tf.get_collection('image_summaries')
    else:
        image_summaries = []
    if not scalar_summaries and not image_summaries:
        return None  # No summaries to write.

    model_dir = params['model_dir']
    iterations_per_loop = params.get('iterations_per_loop', 100)
    img_steps = params['img_summary_steps']

    def host_call_fn(global_step, *args):
        """Training host call. Creates summaries for training metrics."""
        gs = global_step[0]
        with tf.summary.create_file_writer(
                model_dir, max_queue=iterations_per_loop).as_default():
            with tf.summary.record_if(True):
                for i, s in enumerate(scalar_summaries):
                    name = s[0]
                    tensor = args[i][0]
                    tf.summary.scalar(name, tensor, step=gs)

            if img_steps:
                with tf.summary.record_if(lambda: tf.math.equal(gs % img_steps, 0)):
                    # Log images every 1k steps.
                    for i, s in enumerate(image_summaries):
                        name = s[0]
                        tensor = args[i + len(scalar_summaries)]
                        tf.summary.image(name, tensor, step=gs)

            return tf.summary.all_v2_summary_ops()

    reshaped_tensors = [tf.reshape(t, [1]) for _, t in scalar_summaries]
    reshaped_tensors += [t for _, t in image_summaries]
    global_step_t = tf.reshape(global_step, [1])
    return host_call_fn, [global_step_t] + reshaped_tensors


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

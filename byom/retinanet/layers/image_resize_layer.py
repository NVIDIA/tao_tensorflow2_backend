import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow.python.layers.utils as conv_utils
from tensorflow.keras.layers import InputSpec



class ImageResizeLayer(keras.layers.Layer):
    '''Resize Images to a specified size
    https://stackoverflow.com/questions/41903928/add-a-resizing-layer-to-a-keras-sequential-model

    # Arguments
        output_dim: Size of output layer width and height
        output_scale: scale compared with input
        data_format: A string,
            one of `channels_first` (default) or `channels_last`.
        mode: A string,
            one of `nearest` (default) or `bilinear`.

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    '''

    def __init__(self, output_dim=(1, 1), output_scale=None, data_format='channels_first', mode="nearest", **kwargs):
        super(ImageResizeLayer, self).__init__(**kwargs)
        self.naive_output_dim = conv_utils.normalize_tuple(output_dim,
                                                           2, 'output_dim')

        self.data_format = data_format
        if isinstance(output_scale, np.ndarray) or isinstance(output_scale, list):
            output_scale = output_scale[0]
        self.naive_output_scale = output_scale
        self.mode = mode        
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.naive_output_scale is not None:
            if self.data_format == 'channels_first':
                self.output_dim = (self.naive_output_scale * input_shape[2],
                                   self.naive_output_scale * input_shape[3])
            elif self.data_format == 'channels_last':
                self.output_dim = (self.naive_output_scale * input_shape[1],
                                   self.naive_output_scale * input_shape[2])
        else:
            self.output_dim = self.naive_output_dim
            
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], self.output_dim[0], self.output_dim[1])
        elif self.data_format == 'channels_last':
            return (input_shape[0], self.output_dim[0], self.output_dim[1], input_shape[3])

    def call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = keras.layers.Permute((2, 3, 1))(inputs)
        output = tf.image.resize(inputs, self.output_dim, self.mode)

        if self.data_format == 'channels_first':
            output = keras.layers.Permute((3, 1, 2))(output)

        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'mode': self.mode}
        base_config = super(ImageResizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

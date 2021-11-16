"""Postprocessing for anchor-based detection."""
from typing import List, Tuple

from absl import logging
import tensorflow as tf

from blocks.processor.postprocessor import Postprocessor
from byom.retinanet.utils import model_utils
from byom.retinanet.model import anchors

T = tf.Tensor  # a shortcut for typing check.
CLASS_OFFSET = 1


def to_list(inputs):
    if isinstance(inputs, dict):
        return [inputs[k] for k in sorted(inputs.keys())]
    if isinstance(inputs, list):
        return inputs
    raise ValueError('Unrecognized inputs : {}'.format(inputs))


class RetinaNetPostprocessor(Postprocessor):
    def __init__(self, params):
        self.params = params

    def generate_detections(
        self,
        cls_outputs,
        box_outputs,
        image_scales,
        image_ids,
        flip=False):

        pass
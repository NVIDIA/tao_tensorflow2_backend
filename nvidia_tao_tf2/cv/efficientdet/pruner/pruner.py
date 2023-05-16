# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""TAO EfficientDet model pruner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from nvidia_tao_tf2.blocks.pruner import Pruner
from nvidia_tao_tf2.cv.efficientdet.utils.helper import load_model

import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611
logger = logging.getLogger(__name__)


class EfficientDetPruner(Pruner):
    """EfficientDet Pruner."""

    def _load_model(self):
        """Load model."""
        self.model = load_model(self.model_path, self.cfg)
        self.excluded_layers = self.model.output_names
        self.excluded_layers.extend(
            ['box-0', 'box-1', 'box-2', 'class-0', 'class-1', 'class-2',
             'fpn_cell_2_fnode7_op_after_combine_4_conv'])

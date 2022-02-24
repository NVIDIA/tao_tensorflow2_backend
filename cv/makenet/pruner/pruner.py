# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO classification model pruner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from blocks.pruner import Pruner
from cv.makenet.utils.helper import load_model

import common.no_warning # noqa pylint: disable=W0611
logger = logging.getLogger(__name__)


class ClassificationPruner(Pruner):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _load_model(self):
        self.model = load_model(self.model_path, self.key)
        self.excluded_layers = self.model.output_names
        self.model.summary()

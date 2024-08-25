# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            ['box-0', 'box-1', 'box-2', 'class-0', 'class-1', 'class-2'])

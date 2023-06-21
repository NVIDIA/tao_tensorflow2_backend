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

"""Utils for processing config file to run classification pipelines."""


def spec_checker(cfg):
    """Hydra config checker."""
    assert cfg.model.input_channels in [1, 3], "Invalid input image dimension."
    assert cfg.model.input_height >= 16, "Image height should be greater than 15 pixels."
    assert cfg.model.input_width >= 16, "Image width should be greater than 15 pixels."
    assert cfg.model.input_image_depth in [8, 16], "Only 8-bit and 16-bit images are supported"
    assert cfg.dataset.num_classes > 1, \
        "Number of classes should be greater than 1. Consider adding a background class."

    assert cfg.prune.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert cfg.prune.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

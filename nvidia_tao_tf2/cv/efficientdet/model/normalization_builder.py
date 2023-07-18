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
# ==============================================================================
"""Common utils."""
import tensorflow as tf


def batch_norm_class(is_training=True):
    """Choose BN based on training phase."""
    if is_training:
        # TODO(fsx950223): use SyncBatchNorm after TF bug is fixed (incorrect nccl
        # all_reduce). See https://github.com/tensorflow/tensorflow/issues/41980
        return tf.keras.layers.BatchNormalization
    return tf.keras.layers.BatchNormalization

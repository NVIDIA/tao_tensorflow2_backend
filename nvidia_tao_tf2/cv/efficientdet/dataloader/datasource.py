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
"""Data Source Class."""
import os


class DataSource:
    """Datasource class."""

    def __init__(self, tfrecord_patterns, image_dirs):
        """Init."""
        assert len(tfrecord_patterns) > 0, "Error!"
        if len(image_dirs) > 0:
            assert len(image_dirs) == len(tfrecord_patterns), "The number of image directories does" \
                "not match the number of TFRecords paths."
        else:
            image_dirs = ['/'] * len(tfrecord_patterns)
        for i in range(len(tfrecord_patterns)):
            if os.path.isdir(tfrecord_patterns[i]):
                tfrecord_patterns[i] = os.path.join(tfrecord_patterns[i], "*.tfrecord")
        self.tfrecord_patterns = list(tfrecord_patterns)
        self.image_dirs = list(image_dirs)

    def __len__(self):
        """Number of tfrecords."""
        return len(self.tfrecord_patterns)

    def __iter__(self):
        """Return iterator."""
        self.n = 0
        return self

    def __next__(self):
        """Get next record."""
        if self.n < len(self.tfrecord_patterns):
            tfr = self.tfrecord_patterns[self.n]
            img_dir = self.image_dirs[self.n]
            self.n += 1
            return tfr, img_dir
        raise StopIteration

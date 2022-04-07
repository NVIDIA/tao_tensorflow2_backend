# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
"""Data Source Class."""
class DataSource:
    def __init__(self, tfrecord_patterns, image_dirs):
        assert len(tfrecord_patterns) > 0, "Error!"
        if len(image_dirs) > 0:
            assert len(image_dirs) == len(tfrecord_patterns), "The number of image directories does" \
                "not match the number of TFRecords paths."
        else:
            image_dirs = ['/'] * len(tfrecord_patterns)
        self.tfrecord_patterns = list(tfrecord_patterns)
        self.image_dirs = list(image_dirs)
        
    def __len__(self):
        return len(self.tfrecord_patterns)
        
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.tfrecord_patterns):
            tfr = self.tfrecord_patterns[self.n]
            img_dir = self.image_dirs[self.n]
            self.n += 1
            return tfr, img_dir
        else:
           raise StopIteration

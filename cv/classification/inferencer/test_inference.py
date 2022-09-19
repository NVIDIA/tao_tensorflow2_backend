# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Test inferencer image loader."""
import numpy as np
import pytest

from cv.classification.inferencer.inferencer import Inferencer


class TestInferencer(Inferencer):
    def __init__(self, input_shape, model_img_mode='RGB',
                 img_mean=None, keep_aspect_ratio=False) -> None:
        self.model_img_mode = model_img_mode
        self._input_shape = input_shape
        self.keep_aspect_ratio = keep_aspect_ratio
        self.img_mean = img_mean
    
    def infer_single(self):
        print('Do nothing')


@pytest.mark.parametrize("input_shape", [(5, 5, 3), (128, 128, 3)])
def test_image_preprocessing(input_shape):
    h, w, c = input_shape
    dummy_image = np.ones(input_shape).astype(np.uint8)

    inferencer = TestInferencer((c, h, w))
    ori, scale, img = inferencer._load_img(img_alt=dummy_image)
    print(f"scale: {scale}")
    print(img)
    print(img.shape)

# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

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

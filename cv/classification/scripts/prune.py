# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the classification TAO model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile
import tensorflow as tf

from common.hydra.hydra_runner import hydra_runner

from cv.classification.config.default_config import ExperimentConfig
from cv.classification.pruner.pruner import ClassificationPruner
from cv.classification.utils.helper import encode_eff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger(__name__)


def run_pruning(cfg):
    """Prune an encrypted Keras model."""

    assert cfg.prune_config.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert cfg.prune_config.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    pruner = ClassificationPruner(cfg)

    # Pruning trained model
    pruned_model = pruner.prune(
        threshold=cfg.prune_config.pruning_threshold,
        excluded_layers=list(cfg.prune_config.excluded_layers))

    # Save the encrypted pruned model
    tmp_saved_model = tempfile.mkdtemp()
    pruned_model.save(tmp_saved_model)
    encode_eff(tmp_saved_model, cfg.prune_config.output_path, cfg.key)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="prune", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet pruning.
    """
    run_pruning(cfg=cfg)


if __name__ == '__main__':
    main()

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the EfficientDet TAO model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import os
import tempfile
import tensorflow as tf

from cv.efficientdet.config.default_config import ExperimentConfig
from cv.efficientdet.config.hydra_runner import hydra_runner
from cv.efficientdet.pruner.pruner import EfficientDetPruner
from cv.efficientdet.utils.helper import dump_eval_json, dump_json, encode_eff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger(__name__)


def run_pruning(cfg):
    """Prune an encrypted Keras model."""

    output_dir = os.path.dirname(cfg.prune_config.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert cfg.prune_config.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert cfg.prune_config.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    pruner = EfficientDetPruner(cfg)

    # Pruning trained model
    pruned_model = pruner.prune(
        threshold=cfg.prune_config.pruning_threshold,
        excluded_layers=list(cfg.prune_config.excluded_layers))
    pruned_model.summary()

    # Save the encrypted pruned model
    tmp_dir = tempfile.mkdtemp()
    dump_json(pruned_model, os.path.join(tmp_dir, 'train_graph.json'))
    dump_eval_json(tmp_dir, eval_graph='eval_graph.json')
    pruned_model.save_weights(os.path.join(tmp_dir, 'prunedckpt'))
    # Convert to EFF
    encode_eff(tmp_dir, cfg.prune_config.output_path, cfg.key, is_pruned=True)
    pruning_ratio = pruned_model.count_params() / pruner.model.count_params()
    logger.info(
        f"Pruning ratio (pruned model / original model): {pruning_ratio}")


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
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the EfficientDet TAO model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile
import tensorflow as tf

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.logging.logging as status_logging
from nvidia_tao_tf2.common.utils import get_model_file_size

from nvidia_tao_tf2.cv.efficientdet.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.efficientdet.pruner.pruner import EfficientDetPruner
from nvidia_tao_tf2.cv.efficientdet.utils.helper import dump_eval_json, dump_json, encode_eff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger(__name__)


@monitor_status(name='efficientdet', mode='pruning')
def run_pruning(cfg):
    """Prune an encrypted Keras model."""
    logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
    output_dir = os.path.dirname(cfg.prune.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert cfg.prune.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert cfg.prune.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    pruner = EfficientDetPruner(cfg)

    # Pruning trained model
    pruned_model = pruner.prune(
        threshold=cfg.prune.threshold,
        excluded_layers=list(cfg.prune.excluded_layers))

    # Save the encrypted pruned model
    tmp_dir = tempfile.mkdtemp()
    dump_json(pruned_model, os.path.join(tmp_dir, 'train_graph.json'))
    dump_eval_json(tmp_dir, eval_graph='eval_graph.json')
    pruned_model.save_weights(os.path.join(tmp_dir, 'prunedckpt'))
    pruned_model.summary()
    # Convert to EFF
    encode_eff(tmp_dir, cfg.prune.output_path, cfg.key, is_pruned=True)
    pruning_ratio = pruned_model.count_params() / pruner.model.count_params()
    logger.info("Pruning ratio (pruned model / original model): %s", pruning_ratio)
    status_logging.get_status_logger().kpi.update(
        {'pruning_ratio': float(pruning_ratio),
         'param_count': pruned_model.count_params(),
         'size': get_model_file_size(cfg.prune.output_path)})


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="prune", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet pruning."""
    run_pruning(cfg=cfg)


if __name__ == '__main__':
    main()

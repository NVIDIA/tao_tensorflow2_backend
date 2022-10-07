# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the EfficientDet TAO model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile
import tensorflow as tf

from common.hydra.hydra_runner import hydra_runner
import common.logging.logging as status_logging
from common.utils import get_model_file_size

from cv.efficientdet.config.default_config import ExperimentConfig
from cv.efficientdet.pruner.pruner import EfficientDetPruner
from cv.efficientdet.utils.helper import dump_eval_json, dump_json, encode_eff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run_pruning(cfg):
    """Prune an encrypted Keras model."""
    # set up status logger
    status_file = os.path.join(cfg.results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )
    s_logger = status_logging.get_status_logger()
    s_logger.write(
        status_level=status_logging.Status.STARTED,
        message="Starting EfficientDet pruning."
    )
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
    # Convert to EFF
    encode_eff(tmp_dir, cfg.prune.output_path, cfg.key, is_pruned=True)
    pruning_ratio = pruned_model.count_params() / pruner.model.count_params()
    print(f"Pruning ratio (pruned model / original model): {pruning_ratio}")
    s_logger.kpi.update(
        {'pruning_ratio': float(pruning_ratio),
         'param_count': pruned_model.count_params(),
         'size': get_model_file_size(cfg.prune.output_path)})
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Pruning finished successfully."
    )


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

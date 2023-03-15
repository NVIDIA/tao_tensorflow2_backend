# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the classification TAO model."""
import logging
import os
import tempfile
import tensorflow as tf

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.logging.logging as status_logging
from nvidia_tao_tf2.common.utils import get_model_file_size
from nvidia_tao_tf2.common.decorators import monitor_status

from nvidia_tao_tf2.cv.classification.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.classification.pruner.pruner import ClassificationPruner
from nvidia_tao_tf2.cv.classification.utils.helper import encode_eff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='classification', mode='pruning')
def run_pruning(cfg):
    """Prune an encrypted Keras model."""
    logger.setLevel(logging.INFO)
    assert cfg.prune.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert cfg.prune.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    pruner = ClassificationPruner(cfg)

    # Pruning trained model
    pruned_model = pruner.prune(
        threshold=cfg.prune.threshold,
        excluded_layers=list(cfg.prune.excluded_layers))

    # Save the encrypted pruned model
    tmp_saved_model = tempfile.mkdtemp()
    pruned_model.save(tmp_saved_model)
    encode_eff(tmp_saved_model, cfg.prune.output_path, cfg.key)
    # Printing out pruned model summary
    logger.info("Model summary of the pruned model:")
    logger.info(pruned_model.summary())

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
    """Wrapper function for classification pruning."""
    run_pruning(cfg=cfg)


if __name__ == '__main__':
    main()

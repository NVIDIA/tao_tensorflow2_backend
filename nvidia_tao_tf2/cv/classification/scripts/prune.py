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

"""Script to prune the classification TAO model."""
import logging
import os
import tempfile
import tensorflow as tf

from nvidia_tao_core.config.classification_tf2.default_config import ExperimentConfig

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.logging.logging as status_logging
from nvidia_tao_tf2.common.utils import get_model_file_size, update_results_dir

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
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir, exist_ok=True)
    pruner = ClassificationPruner(cfg)

    # Pruning trained model
    pruned_model = pruner.prune(
        threshold=cfg.prune.threshold,
        excluded_layers=list(cfg.prune.excluded_layers))

    # Save the encrypted pruned model
    tmp_saved_model = tempfile.mkdtemp()
    pruned_model.save(tmp_saved_model)
    # Convert to EFF
    output_path = os.path.join(
        cfg.prune.results_dir,
        f'model_th={cfg.prune.threshold}_eq={cfg.prune.equalization_criterion}.tlt')
    encode_eff(tmp_saved_model, output_path, cfg.encryption_key)
    # Printing out pruned model summary
    logger.info("Model summary of the pruned model:")
    logger.info(pruned_model.summary())

    pruning_ratio = pruned_model.count_params() / pruner.model.count_params()
    logger.info("Pruning ratio (pruned model / original model): %s", pruning_ratio)
    status_logging.get_status_logger().kpi.update(
        {'pruning_ratio': float(pruning_ratio),
         'param_count': pruned_model.count_params(),
         'size': get_model_file_size(output_path)})


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="prune", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for classification pruning."""
    cfg = update_results_dir(cfg, 'prune')
    run_pruning(cfg=cfg)


if __name__ == '__main__':
    main()

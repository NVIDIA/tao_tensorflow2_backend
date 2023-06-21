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

"""Script to prune the EfficientDet TAO model."""

import logging
import os
import tempfile
import tensorflow as tf

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.logging.logging as status_logging
from nvidia_tao_tf2.common.utils import get_model_file_size, update_results_dir

from nvidia_tao_tf2.cv.efficientdet.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.efficientdet.pruner.pruner import EfficientDetPruner
from nvidia_tao_tf2.cv.efficientdet.utils.helper import dump_eval_json, dump_json, encode_eff
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import initialize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='efficientdet', mode='pruning')
def run_pruning(cfg):
    """Prune an encrypted Keras model."""
    # Set up EfficientDet pruner
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
    output_path = os.path.join(
        cfg.prune.results_dir,
        f'model_th={cfg.prune.threshold}_eq={cfg.prune.equalization_criterion}.tlt')
    encode_eff(tmp_dir, output_path, cfg.encryption_key, is_pruned=True)
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
    """Wrapper function for EfficientDet pruning."""
    cfg = update_results_dir(cfg, 'prune')
    initialize(cfg, logger, training=False)
    run_pruning(cfg=cfg)


if __name__ == '__main__':
    main()

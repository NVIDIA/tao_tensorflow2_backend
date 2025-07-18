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

"""Export a classification model."""
import os
import logging
import sys

from nvidia_tao_core.config.classification_tf2.default_config import ExperimentConfig

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.utils import update_results_dir

from nvidia_tao_tf2.cv.classification.export.classification_exporter import Exporter
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='classification', mode='export')
def run_export(cfg=None):
    """Export classification model to etlt."""
    logger.setLevel(logging.INFO)
    # Deprecated: DLFW 25.01 doesn't support tensorflow_quantization
    if sys.version_info >= (3, 12):
        logger.warning("DeprecationWarning: QAT is not supported after DLFW 25.01. Using normal training.")
        cfg.train.qat = False

    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir, exist_ok=True)
    exporter = Exporter(config=cfg)
    exporter.export()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="export", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application."""
    cfg = update_results_dir(cfg, 'export')
    run_export(cfg=cfg)


if __name__ == "__main__":
    main()

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
"""The main training script."""
import logging
import os
import sys

from nvidia_tao_core.config.efficientdet_tf2.default_config import ExperimentConfig

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_tf2.common.mlops.utils import init_mlops
import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf2.common.utils import update_results_dir

from nvidia_tao_tf2.cv.efficientdet.dataloader import dataloader, datasource
from nvidia_tao_tf2.cv.efficientdet.model.efficientdet_module import EfficientDetModule
from nvidia_tao_tf2.cv.efficientdet.model import callback_builder
from nvidia_tao_tf2.cv.efficientdet.trainer.efficientdet_trainer import EfficientDetTrainer
from nvidia_tao_tf2.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import is_main_process, initialize
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import get_world_size, get_rank
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='efficientdet', mode='training')
def run_experiment(cfg):
    """Run training experiment."""
    # Deprecated: DLFW 25.01 doesn't support tensorflow_quantization
    if sys.version_info >= (3, 12):
        logger.warning("DeprecationWarning: QAT is not supported after DLFW 25.01. Using normal training.")
        cfg.train.qat = False

    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode='train'))

    if is_main_process():
        init_mlops(cfg, name='efficientdet')

    # Set up dataloader
    train_sources = datasource.DataSource(
        cfg.dataset.train_tfrecords,
        cfg.dataset.train_dirs)
    train_dl = dataloader.CocoDataset(
        train_sources,
        is_training=True,
        use_fake_data=cfg.dataset.use_fake_data,
        max_instances_per_image=config.max_instances_per_image)
    train_dataset = train_dl(
        config.as_dict(),
        batch_size=cfg.train.batch_size)
    # eval data
    eval_sources = datasource.DataSource(
        cfg.dataset.val_tfrecords,
        cfg.dataset.val_dirs)
    eval_dl = dataloader.CocoDataset(
        eval_sources,
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)
    eval_dataset = eval_dl(
        config.as_dict(),
        batch_size=cfg.evaluate.batch_size)

    efficientdet = EfficientDetModule(config)
    # set up callbacks
    callbacks = callback_builder.get_callbacks(
        config,
        eval_dataset.shard(get_world_size(), get_rank()).take(config.eval_samples),
        efficientdet.steps_per_epoch,
        eval_model=efficientdet.eval_model,
        initial_epoch=efficientdet.initial_epoch)
    trainer = EfficientDetTrainer(
        num_epochs=config.num_epochs,
        callbacks=callbacks)
    trainer.fit(
        efficientdet,
        train_dataset,
        eval_dataset,
        verbose=1 if is_main_process() else 0)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet training."""
    cfg = update_results_dir(cfg, 'train')
    initialize(cfg, logger, training=True)
    run_experiment(cfg=cfg)


if __name__ == '__main__':
    main()

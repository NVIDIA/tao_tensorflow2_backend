# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""The main training script."""
import logging
import os

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_tf2.common.mlops.utils import init_mlops
import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611

from nvidia_tao_tf2.cv.efficientdet.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.efficientdet.dataloader import dataloader, datasource
from nvidia_tao_tf2.cv.efficientdet.model.efficientdet_module import EfficientDetModule
from nvidia_tao_tf2.cv.efficientdet.model import callback_builder
from nvidia_tao_tf2.cv.efficientdet.trainer.efficientdet_trainer import EfficientDetTrainer
from nvidia_tao_tf2.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import is_main_process, initialize
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import get_world_size, get_rank
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)


def setup_env(cfg):
    """Setup training env."""
    # hvd.init()
    logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
    # initialize
    initialize(cfg, training=True)
    if is_main_process():
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir)
        init_mlops(cfg, name='efficientdet')


@monitor_status(name='efficientdet', mode='training')
def run_experiment(cfg):
    """Run training experiment."""
    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode='train'))

    # Set up dataloader
    train_sources = datasource.DataSource(
        cfg.data.train_tfrecords,
        cfg.data.train_dirs)
    train_dl = dataloader.CocoDataset(
        train_sources,
        is_training=True,
        use_fake_data=cfg.data.use_fake_data,
        max_instances_per_image=config.max_instances_per_image)
    train_dataset = train_dl(
        config.as_dict(),
        batch_size=cfg.train.batch_size)
    # eval data
    eval_sources = datasource.DataSource(
        cfg.data.val_tfrecords,
        cfg.data.val_dirs)
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
        config.num_epochs,
        verbose=1 if is_main_process() else 0)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet training."""
    setup_env(cfg)
    run_experiment(cfg=cfg)


if __name__ == '__main__':
    main()

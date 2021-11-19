"""The main training script."""
import os
import time
from mpi4py import MPI
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger

from byom.retinanet.config.hydra_runner import hydra_runner
from byom.retinanet.config.default_config import ExperimentConfig
from byom.retinanet.dataloader import dataloader
from byom.retinanet.losses import losses
from byom.retinanet.model.retinanet import retinanet
from byom.retinanet.model import callback_builder
from byom.retinanet.model import optimizer_builder
from byom.retinanet.trainer.retinanet_trainer import RetinaNetTrainer
from byom.retinanet.utils.config_utils import generate_params_from_cfg
from byom.retinanet.utils import hparams_config
from byom.retinanet.utils.horovod_utils import is_main_process, get_world_size, get_rank, initialize


def run_experiment(cfg, results_dir, key):

    # get e2e training time
    begin = time.time()
    logging.info("Training started at: {}".format(time.asctime()))

    hvd.init()

    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg['model_config']['model_name'])
    config.update(generate_params_from_cfg(config, cfg, mode='train'))

    # initialize
    initialize(config, training=True)
    # dllogger setup
    backends = []
    if is_main_process():
        log_path = os.path.join(results_dir, 'log.txt')
        backends += [
            JSONStreamBackend(verbosity=Verbosity.VERBOSE, filename=log_path),
            StdOutBackend(verbosity=Verbosity.DEFAULT)]
    DLLogger.init(backends=backends)

    steps_per_epoch = (cfg['train_config']['num_examples_per_epoch'] + 
        (cfg['train_config']['train_batch_size'] * get_world_size()) - 1) // \
            (cfg['train_config']['train_batch_size'] * get_world_size())

    # Set up dataloader
    train_reader = dataloader.InputReader(
        cfg['data_config']['training_file_pattern'],
        is_training=True,
        use_fake_data=cfg['data_config']['use_fake_data'],
        max_instances_per_image=config.max_instances_per_image)
    train_dataset = train_reader(
        config.as_dict(),
        batch_size=cfg['train_config']['train_batch_size'])

    eval_reader = dataloader.InputReader(
        cfg['data_config']['validation_file_pattern'],
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)
    eval_dataset = eval_reader(
        config.as_dict(),
        batch_size=cfg['eval_config']['eval_batch_size'])

    # Compile model
    # pick focal loss implementation
    focal_loss = losses.StableFocalLoss(
        config.alpha,
        config.gamma,
        label_smoothing=config.label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE)
    input_shape = [3, 512, 512]
    
    model = retinanet(
        input_shape,
        model_path=cfg['model_config']['train_graph'],
        training=True)
    model.summary()
    model.compile(
        optimizer=optimizer_builder.get_optimizer(cfg['train_config']),
        loss={
            'box_loss':
                losses.BoxLoss(
                    0.1, # config['delta'],
                    reduction=tf.keras.losses.Reduction.NONE),
            'box_iou_loss':
                losses.BoxIouLoss(
                    config.iou_loss_type,
                    config.min_level,
                    config.max_level,
                    config.num_scales,
                    config.aspect_ratios,
                    config.anchor_scale,
                    config.image_size,
                    reduction=tf.keras.losses.Reduction.NONE),
            'class_loss': focal_loss,
            'seg_loss':
                tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
        })

    callbacks = callback_builder.get_callbacks(
        cfg, 'traineval', eval_dataset, 
        DLLogger,
        profile=False, 
        time_history=False,
        log_steps=1,
        lr_tb=False,
        benchmark=False)

    # TODO: get RetinaNetTrainer from builder+config
    trainer = RetinaNetTrainer(model, config, callbacks)
    trainer.fit(
        train_dataset,
        eval_dataset,
        config.num_epochs,
        steps_per_epoch,
        initial_epoch=0,
        validation_steps=500,
        verbose=1 if is_main_process() else 0)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of BYOM application.
    """
    run_experiment(cfg=cfg,
                   results_dir=cfg.results_dir,
                   key=cfg.key)


if __name__ == '__main__':
    main()
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""The main training script."""
import os
import sys
import time
from absl import logging
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase
from tensorflow_quantization.quantize import quantize_model

from common.hydra.hydra_runner import hydra_runner
import common.logging.logging as status_logging

from cv.efficientdet.config.default_config import ExperimentConfig
from cv.efficientdet.dataloader import dataloader, datasource
from cv.efficientdet.losses import losses
from cv.efficientdet.model.efficientdet import efficientdet
from cv.efficientdet.model import callback_builder
from cv.efficientdet.model import optimizer_builder
from cv.efficientdet.trainer.efficientdet_trainer import EfficientDetTrainer
from cv.efficientdet.utils import hparams_config, keras_utils
from cv.efficientdet.utils.config_utils import generate_params_from_cfg
from cv.efficientdet.utils.helper import decode_eff, dump_json, load_model, load_json_model
from cv.efficientdet.utils.horovod_utils import is_main_process, get_world_size, get_rank, initialize


def run_experiment(cfg, ci_run=False):
    """Run training experiment."""
    # get e2e training time
    # begin = time.time()
    logging.info(f"Training started at: {time.asctime()}")

    hvd.init()

    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode='train'))

    # initialize
    initialize(config, training=True, ci_run=ci_run)

    if is_main_process():
        if not os.path.exists(cfg.train.results_dir):
            os.makedirs(cfg.train.results_dir)

    steps_per_epoch = (
        cfg.train.num_examples_per_epoch +
        (cfg.train.batch_size * get_world_size()) - 1) // \
        (cfg.train.batch_size * get_world_size())

    # set up status logger
    status_file = os.path.join(cfg.train.results_dir, "status.json")
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
        message="Starting EfficientDet training."
    )

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

    # Build models
    if not cfg.train.pruned_model_path:
        if is_main_process():
            print("Building unpruned graph...")
        # TODO(@yuw): verify channels_first training or force last
        input_shape = list(config.image_size) + [3] \
            if config.data_format == 'channels_last' else [3] + list(config.image_size)
        original_learning_phase = tf.keras.backend.learning_phase()
        model = efficientdet(input_shape, training=True, config=config)
        tf.keras.backend.set_learning_phase(0)
        eval_model = efficientdet(input_shape, training=False, config=config)
        tf.keras.backend.set_learning_phase(original_learning_phase)
    else:
        if is_main_process():
            print("Loading pruned graph...")
        original_learning_phase = tf.keras.backend.learning_phase()
        model = load_model(cfg.train.pruned_model_path, cfg, mode='train')
        tf.keras.backend.set_learning_phase(0)
        eval_model = load_model(cfg.train.pruned_model_path, cfg, mode='eval')
        tf.keras.backend.set_learning_phase(original_learning_phase)

    # save nonQAT nonAMP graph in results_dir
    dump_json(model, os.path.join(cfg.train.results_dir, 'train_graph.json'))
    dump_json(eval_model, os.path.join(cfg.train.results_dir, 'eval_graph.json'))

    # Load pretrained weights
    # TODO(@yuw): move weight loading to master rank
    resume_ckpt_path = os.path.join(cfg.train.results_dir, f'{config.name}.resume')
    if str(cfg.train.checkpoint).endswith(".tlt"):
        pretrained_ckpt_path, ckpt_name = decode_eff(str(cfg.train.checkpoint), cfg.key)
    else:
        pretrained_ckpt_path = cfg.train.checkpoint
    if pretrained_ckpt_path and not os.path.exists(resume_ckpt_path):
        if not cfg.train.pruned_model_path:
            print("Loading pretrained weight...")
            if 'train_graph.json' in os.listdir(pretrained_ckpt_path):
                print("Loading EfficientDet mdoel...")
                pretrained_model = load_json_model(
                    os.path.join(pretrained_ckpt_path, 'train_graph.json'))
                keras_utils.restore_ckpt(
                    pretrained_model,
                    os.path.join(pretrained_ckpt_path, ckpt_name),
                    cfg.train.moving_average_decay,
                    steps_per_epoch=0,
                    expect_partial=True)
            else:
                print("Loading EfficientNet backbone...")
                pretrained_model = tf.keras.models.load_model(pretrained_ckpt_path)
            for layer in pretrained_model.layers[1:]:
                # The layer must match up to prediction layers.
                l_return = None
                try:
                    l_return = model.get_layer(layer.name)
                except ValueError:
                    # Some layers are not there
                    print(f"Skipping as {layer.name} is not found.")
                if l_return is not None:
                    try:
                        l_return.set_weights(layer.get_weights())
                    except ValueError:
                        print(f"Skipping {layer.name} due to shape mismatch.")

    # TODO(@yuw): Enable QAT
    if cfg.train.qat:
        model = quantize_model(model, custom_qdq_cases=[EfficientNetQDQCase()])
        eval_model = quantize_model(eval_model, custom_qdq_cases=[EfficientNetQDQCase()])

    if is_main_process():
        model.summary()
    # Compile model
    # pick focal loss implementation
    focal_loss = losses.StableFocalLoss(
        config.alpha,
        config.gamma,
        label_smoothing=config.label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE)
    model.compile(
        optimizer=optimizer_builder.get_optimizer(cfg.train, steps_per_epoch),
        loss={
            'box_loss':
                losses.BoxLoss(
                    config.delta,  # TODO(@yuw): add to default config
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

    # resume from checkpoint or load pretrained backbone
    train_from_epoch = 0
    if os.path.exists(resume_ckpt_path):
        if is_main_process():
            print("Resume training...")
        ckpt_path, _ = decode_eff(resume_ckpt_path, cfg.key)
        train_from_epoch = keras_utils.restore_ckpt(
            model,
            ckpt_path,
            config.moving_average_decay,
            steps_per_epoch=steps_per_epoch,
            expect_partial=True)
        # TODO(@yuw): remove ckpt_path

    if train_from_epoch < config.num_epochs:
        # set up callbacks
        num_samples = (cfg.evaluate.num_samples + get_world_size() - 1) // get_world_size()
        num_samples = (num_samples + cfg.evaluate.batch_size - 1) // cfg.evaluate.batch_size
        cfg.evaluate.num_samples = num_samples

        callbacks = callback_builder.get_callbacks(
            cfg,
            eval_dataset.shard(get_world_size(), get_rank()).take(num_samples),
            steps_per_epoch,
            eval_model=eval_model,
            initial_epoch=train_from_epoch)

        trainer = EfficientDetTrainer(model, config, callbacks)
        trainer.fit(
            train_dataset,
            eval_dataset,
            config.num_epochs,
            steps_per_epoch,
            initial_epoch=train_from_epoch,
            validation_steps=num_samples // cfg.evaluate.batch_size,
            verbose=1 if is_main_process() else 0)
    else:
        if is_main_process():
            print(f"Training ({train_from_epoch} epochs) has finished.")
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet training."""
    try:
        run_experiment(cfg=cfg)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == '__main__':
    main()

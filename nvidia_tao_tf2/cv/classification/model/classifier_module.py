# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""The main training script."""
import logging
import os

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase, ResNetV1QDQCase
from tensorflow_quantization.quantize import quantize_model

from nvidia_tao_tf2.blocks.module import TAOModule
from nvidia_tao_tf2.cv.classification.model.model_builder import get_model
from nvidia_tao_tf2.cv.classification.utils.helper import (
    build_optimizer,
    decode_tltb,
    load_model,
    setup_config)
logger = logging.getLogger(__name__)


class ClassifierModule(TAOModule):
    """TAO Classifer Module."""

    def __init__(self, cfg, steps_per_epoch):
        """Init."""
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.pretrained_model = None
        self.model = self._build_models(cfg)

        self.initial_epoch, ckpt_path = self._get_latest_checkpoint(
            cfg.results_dir, self.model.name)

        self._load_pretrained_weights(ckpt_path, cfg)

        self.configure_optimizers(cfg)
        self.configure_losses(cfg)
        self._quantize_models(cfg)
        self.compile()
        if hvd.rank() == 0:
            self.model.summary()

    def _quantize_models(self, cfg):
        """Quantize models."""
        if cfg.train.qat:
            logger.info("QAT enabled.")
            qdq_cases = [EfficientNetQDQCase()] \
                if 'efficientnet' in cfg.model.backbone else [ResNetV1QDQCase()]
            self.model = quantize_model(self.model, custom_qdq_cases=qdq_cases)

    def _build_models(self, cfg):
        """Build classification model."""
        if cfg['data_format'] == 'channels_first':
            input_shape = (cfg.model.input_channels, cfg.model.input_height, cfg.model.input_width)
        else:
            input_shape = (cfg.model.input_height, cfg.model.input_width, cfg.model.input_channels)

        ka = {
            'use_batch_norm': cfg['model']['use_batch_norm'],
            'use_pooling': cfg['model']['use_pooling'],
            'freeze_bn': cfg['model']['freeze_bn'],
            'use_bias': cfg['model']['use_bias'],
            'all_projections': cfg['model']['all_projections'],
            'dropout': cfg['model']['dropout'],
            'model_config_path': cfg['model']['byom_model'],
            'passphrase': cfg['encryption_key'],
            'activation_type': cfg['model']['activation_type'],
        }
        model = get_model(
            backbone=cfg['model']['backbone'],
            input_shape=input_shape,
            data_format=cfg['data_format'],
            nclasses=cfg['dataset']['num_classes'],
            retain_head=cfg['model']['retain_head'],
            freeze_blocks=cfg['model']['freeze_blocks'],
            **ka)
        # @scha: Load CUSTOM_OBJS from BYOM
        if cfg['model']['backbone'] in ["byom"]:
            custom_objs = decode_tltb(ka['model_config_path'], ka['passphrase'])['custom_objs']
        else:
            custom_objs = {}

        # Set up BN and regularizer config
        model = setup_config(
            model,
            cfg.train.reg_config,
            bn_config=cfg.train.bn_config,
            custom_objs=custom_objs
        )
        return model

    def configure_losses(self, cfg, loss=None):
        """Configure losses."""
        self.losses = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.train.label_smoothing)

    def configure_optimizers(self, cfg):
        """Configure optimizers."""
        if self.initial_epoch:
            self.model = self.pretrained_model
            opt = self.pretrained_model.optimizer
        else:
            # Defining optimizer
            opt = build_optimizer(cfg.train.optim_config)
        # Add Horovod Distributed Optimizer
        self.opt = hvd.DistributedOptimizer(
            opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    def compile(self):
        """Compile model."""
        self.model.compile(
            loss=self.losses,
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            optimizer=self.opt,
            experimental_run_tf_function=False)

    def _load_pretrained_weights(self, ckpt_path, cfg):
        """Load pretrained weights."""
        ckpt_path = ckpt_path or cfg.train.checkpoint
        if ckpt_path:
            if hvd.rank() == 0:
                logger.info("Loading pretrained model from %s", ckpt_path)
            # Decrypt and load pretrained model
            self.pretrained_model = load_model(
                ckpt_path,
                passphrase=cfg.encryption_key)

            strict_mode = True
            for layer in self.pretrained_model.layers[1:]:
                # The layer must match up to prediction layers.
                if 'predictions' in layer.name:
                    strict_mode = False
                try:
                    l_return = self.model.get_layer(layer.name)
                except ValueError:
                    # Some layers are not there
                    continue
                try:
                    l_return.set_weights(layer.get_weights())
                except ValueError:
                    if strict_mode:
                        # This is a pruned model
                        self.model = setup_config(
                            self.pretrained_model,
                            cfg.train.reg_config,
                            bn_config=cfg.train.bn_config
                        )

    def _get_latest_checkpoint(self, model_dir, model_name='efficientnet-b'):
        """Get the last tlt checkpoint."""
        if not os.path.exists(model_dir):
            return 0, None
        last_checkpoint = ''
        for f in os.listdir(model_dir):
            if f.startswith(model_name) and f.endswith('.tlt'):
                last_checkpoint = last_checkpoint if last_checkpoint > f else f
        if not last_checkpoint:
            return 0, None
        initial_epoch = int(last_checkpoint[:-4].split('_')[-1])
        if hvd.rank() == 0:
            logger.info('Resume training from #%d epoch', initial_epoch)
        return initial_epoch, os.path.join(model_dir, last_checkpoint)

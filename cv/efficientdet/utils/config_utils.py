# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Utils for processing config file to run EfficientDet training, evaluation, pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import six
from cv.efficientdet.utils import model_utils
logger = logging.getLogger(__name__)


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def generate_params_from_cfg(default_hparams, cfg, mode):
    """Generate parameters from experient cfg."""
    if cfg['model_config']['aspect_ratios']:
        aspect_ratios = eval_str(cfg['model_config']['aspect_ratios'])
        if not isinstance(aspect_ratios, list):
            raise SyntaxError("aspect_ratios should be a list of tuples.")
    else:
        aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    return dict(
        default_hparams.as_dict(),
        # model_config
        name=cfg['model_config']['model_name'],
        aspect_ratios=aspect_ratios,
        anchor_scale=cfg['model_config']['anchor_scale'] or 4,
        min_level=cfg['model_config']['min_level'] or 3,
        max_level=cfg['model_config']['max_level'] or 7,
        num_scales=cfg['model_config']['num_scales'] or 3,
        freeze_bn=cfg['model_config']['freeze_bn'],
        freeze_blocks=cfg['model_config']['freeze_blocks']
        if cfg['model_config']['freeze_blocks'] else None,
        # data config
        val_json_file=cfg['data_config']['validation_json_file'],
        testdev_dir=cfg['data_config']['testdev_dir'],
        num_classes=cfg['data_config']['num_classes'],
        max_instances_per_image=cfg['data_config']['max_instances_per_image'] or 100,
        skip_crowd_during_training=cfg['data_config']['skip_crowd_during_training'],
        # Parse image size in case it is in string format. (H, W)
        image_size=model_utils.parse_image_size(cfg['data_config']['image_size']),
        # augmentation config
        input_rand_hflip=cfg['augmentation_config']['rand_hflip'],
        jitter_min=cfg['augmentation_config']['random_crop_min_scale'] or 0.1,
        jitter_max=cfg['augmentation_config']['random_crop_max_scale'] or 2.0,
        # train eval config
        momentum=cfg['train_config']['momentum'] or 0.9,
        iterations_per_loop=cfg['train_config']['iterations_per_loop'],
        num_examples_per_epoch=cfg['train_config']['num_examples_per_epoch'],
        checkpoint=cfg['train_config']['checkpoint'],
        ckpt=None,
        mode=mode,
        is_training_bn=mode=='train',
        checkpoint_period=cfg['train_config']['checkpoint_period'],
        train_batch_size=cfg['train_config']['train_batch_size'],
        learning_rate=cfg['train_config']['learning_rate'],
        tf_random_seed=cfg['train_config']['tf_random_seed'] or 42,
        pruned_model_path=cfg['train_config']['pruned_model_path'],
        moving_average_decay=cfg['train_config']['moving_average_decay'],
        lr_warmup_epoch=cfg['train_config']['lr_warmup_epoch'] or 5,
        lr_warmup_init=cfg['train_config']['lr_warmup_init'] or 0.00001,
        amp=cfg['train_config']['amp'],
        mixed_precision=cfg['train_config']['amp'] and not cfg['train_config']['qat'], #TODO(@yuw): whether raise error when qat and amp both True?
        data_format=cfg['data_format'],
        l2_weight_decay=cfg['train_config']['l2_weight_decay'],
        l1_weight_decay=cfg['train_config']['l1_weight_decay'],
        clip_gradients_norm=cfg['train_config']['clip_gradients_norm'] or 5.0,
        skip_checkpoint_variables=cfg['train_config']['skip_checkpoint_variables'],
        num_epochs=cfg['train_config']['num_epochs'],
        lr_decay_method=cfg['train_config']['lr_decay_method'],
        # eval config
        eval_epoch_cycle=cfg['eval_config']['eval_epoch_cycle'],
        eval_batch_size=cfg['eval_config']['eval_batch_size'],
        eval_samples=cfg['eval_config']['eval_samples'],
        #
        results_dir=cfg['results_dir']
    )

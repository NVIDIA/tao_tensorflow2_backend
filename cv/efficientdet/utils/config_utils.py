# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Utils for processing config file to run EfficientDet training, evaluation, pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
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
    if cfg['model']['aspect_ratios']:
        aspect_ratios = eval_str(cfg['model']['aspect_ratios'])
        if not isinstance(aspect_ratios, list):
            raise SyntaxError("aspect_ratios should be a list of tuples.")
    else:
        aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    return dict(
        default_hparams.as_dict(),
        # model
        name=cfg['model']['name'],
        model_name=cfg['model']['name'],
        aspect_ratios=aspect_ratios,
        anchor_scale=cfg['model']['anchor_scale'] or 4,
        min_level=cfg['model']['min_level'] or 3,
        max_level=cfg['model']['max_level'] or 7,
        num_scales=cfg['model']['num_scales'] or 3,
        freeze_bn=cfg['model']['freeze_bn'],
        freeze_blocks=cfg['model']['freeze_blocks']
        if cfg['model']['freeze_blocks'] else None,
        # data config
        val_json_file=cfg['data']['val_json_file'],
        testdev_dir=cfg['data']['testdev_dir'],
        num_classes=cfg['data']['num_classes'],
        max_instances_per_image=cfg['data']['max_instances_per_image'] or 100,
        skip_crowd_during_training=cfg['data']['skip_crowd_during_training'],
        # Parse image size in case it is in string format. (H, W)
        image_size=model_utils.parse_image_size(cfg['data']['image_size']),
        # Loader config
        shuffle_file=cfg['data']['loader']['shuffle_file'],
        shuffle_buffer=cfg['data']['loader']['shuffle_buffer'] or 1024,
        cycle_length=cfg['data']['loader']['cycle_length'] or 32,
        block_length=cfg['data']['loader']['block_length'] or 16,
        prefetch_size=cfg['data']['loader']['prefetch_size'] or 2,  # set to 0 for AUTOTUNE
        # augmentation config
        input_rand_hflip=cfg['augment']['rand_hflip'],
        jitter_min=cfg['augment']['random_crop_min_scale'] or 0.1,
        jitter_max=cfg['augment']['random_crop_max_scale'] or 2.0,
        # train config
        iterations_per_loop=cfg['train']['iterations_per_loop'],
        num_examples_per_epoch=cfg['train']['num_examples_per_epoch'],
        checkpoint=cfg['train']['checkpoint'],
        ckpt=None,
        mode=mode,
        is_training_bn=mode == 'train',
        checkpoint_period=cfg['train']['checkpoint_period'],
        train_batch_size=cfg['train']['batch_size'],
        tf_random_seed=cfg['train']['tf_random_seed'] or 42,
        pruned_model_path=cfg['train']['pruned_model_path'],
        moving_average_decay=cfg['train']['moving_average_decay'],
        amp=cfg['train']['amp'],
        mixed_precision=cfg['train']['amp'] and not cfg['train']['qat'],  # TODO(@yuw): whether raise error when qat and amp both True?
        data_format=cfg['data_format'],
        l2_weight_decay=cfg['train']['l2_weight_decay'],
        l1_weight_decay=cfg['train']['l1_weight_decay'],
        clip_gradients_norm=cfg['train']['clip_gradients_norm'] or 5.0,
        skip_checkpoint_variables=cfg['train']['skip_checkpoint_variables'],
        num_epochs=cfg['train']['num_epochs'],
        # LR config
        lr_decay_method=cfg['train']['lr_schedule']['name'],
        learning_rate=cfg['train']['lr_schedule']['learning_rate'],
        lr_warmup_epoch=cfg['train']['lr_schedule']['warmup_epoch'] or 5,
        lr_warmup_init=cfg['train']['lr_schedule']['warmup_init'] or 0.00001,
        # Optimizer config
        momentum=cfg['train']['optimizer']['momentum'] or 0.9,
        optimizer=cfg['train']['optimizer']['name'] or 'sgd',
        # eval config
        eval_epoch_cycle=cfg['evaluate']['eval_epoch_cycle'],
        eval_batch_size=cfg['evaluate']['batch_size'],
        eval_samples=cfg['evaluate']['num_samples'],
        #
        results_dir=cfg['train']['results_dir']
    )

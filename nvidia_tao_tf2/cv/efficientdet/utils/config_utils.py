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

"""Utils for processing config file to run EfficientDet pipelines."""
import six


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""
    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def generate_params_from_cfg(default_hparams, cfg, mode):
    """Generate parameters from experient cfg."""
    spec_checker(cfg)
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
        image_size=(cfg['model']['input_height'], cfg['model']['input_width']),
        # data config
        val_json_file=cfg['dataset']['val_json_file'],
        num_classes=cfg['dataset']['num_classes'],
        max_instances_per_image=cfg['dataset']['max_instances_per_image'] or 100,
        skip_crowd_during_training=cfg['dataset']['skip_crowd_during_training'],
        # Loader config
        shuffle_file=cfg['dataset']['loader']['shuffle_file'],
        shuffle_buffer=cfg['dataset']['loader']['shuffle_buffer'] or 1024,
        cycle_length=cfg['dataset']['loader']['cycle_length'] or 32,
        block_length=cfg['dataset']['loader']['block_length'] or 16,
        prefetch_size=cfg['dataset']['loader']['prefetch_size'] or 2,  # set to 0 for AUTOTUNE
        # augmentation config
        input_rand_hflip=cfg['dataset']['augmentation']['rand_hflip'],
        jitter_min=cfg['dataset']['augmentation']['random_crop_min_scale'] or 0.1,
        jitter_max=cfg['dataset']['augmentation']['random_crop_max_scale'] or 2.0,
        auto_color=cfg['dataset']['augmentation']['auto_color_distortion'],
        auto_translate_xy=cfg['dataset']['augmentation']['auto_translate_xy'],
        auto_augment=cfg['dataset']['augmentation']['auto_color_distortion'] or cfg['dataset']['augmentation']['auto_translate_xy'],
        # train config
        num_examples_per_epoch=cfg['train']['num_examples_per_epoch'],
        checkpoint=cfg['train']['checkpoint'],
        resume_training_checkpoint_path=cfg['train']['resume_training_checkpoint_path'],
        mode=mode,
        is_training_bn=mode == 'train',
        checkpoint_interval=cfg['train']['checkpoint_interval'],
        validation_interval=cfg['train']['validation_interval'],
        train_batch_size=cfg['train']['batch_size'],
        seed=cfg['train']['random_seed'] or 42,
        pruned_model_path=cfg['train']['pruned_model_path'],
        moving_average_decay=cfg['train']['moving_average_decay'],
        mixed_precision=cfg['train']['amp'] and not cfg['train']['qat'],
        qat=cfg['train']['qat'],
        data_format=cfg['data_format'],
        l2_weight_decay=cfg['train']['l2_weight_decay'],
        l1_weight_decay=cfg['train']['l1_weight_decay'],
        clip_gradients_norm=cfg['train']['clip_gradients_norm'] or 5.0,
        skip_checkpoint_variables=cfg['train']['skip_checkpoint_variables'],
        num_epochs=cfg['train']['num_epochs'],
        image_preview=cfg['train']['image_preview'],
        init_epoch=cfg['train']['init_epoch'],
        # LR config
        lr_decay_method=cfg['train']['lr_schedule']['name'],
        learning_rate=cfg['train']['lr_schedule']['learning_rate'],
        lr_warmup_epoch=cfg['train']['lr_schedule']['warmup_epoch'] or 5,
        lr_warmup_init=cfg['train']['lr_schedule']['warmup_init'] or 0.00001,
        lr_annealing_epoch=cfg['train']['lr_schedule']['annealing_epoch'] or 10,
        # Optimizer config
        momentum=cfg['train']['optimizer']['momentum'] or 0.9,
        optimizer=cfg['train']['optimizer']['name'] or 'sgd',
        # eval config
        eval_batch_size=cfg['evaluate']['batch_size'],
        eval_samples=cfg['evaluate']['num_samples'],
        eval_start=cfg['evaluate']['start_eval_epoch'],
        eval_label_map=cfg['evaluate']['label_map'],
        eval_sigma=cfg['evaluate']['sigma'],
        max_output_size=cfg['evaluate']['max_detections_per_image'],
        max_nms_inputs=cfg['evaluate']['max_nms_inputs'],
        #
        results_dir=cfg['results_dir'],
        encryption_key=cfg['encryption_key'],
        iou_loss_type=cfg['train']['iou_loss_type'],
        box_loss_weight=cfg['train']['box_loss_weight'],
        iou_loss_weight=cfg['train']['iou_loss_weight'],
        label_smoothing=cfg['train']['label_smoothing'],
    )


def spec_checker(cfg):
    """Check if parameters in the spec file are valid.

    Args:
        cfg: Hydra config.
    """
    assert cfg.data_format == 'channels_last', "Only `channels_last` data format is supported."
    # training config
    assert cfg.train.batch_size > 0, \
        "batch size for training must be positive."
    assert cfg.train.num_examples_per_epoch > 0, \
        "Number of samples must be positive."
    # Check validation and checkpoint intervals against total number of epochs.
    assert 0 < cfg.train.checkpoint_interval <= cfg.train.num_epochs, (
        f"Checkpoint interval is not between 0 and {cfg.train.num_epochs}: {cfg.train.checkpoint_interval}"
    )
    assert 0 < cfg.train.validation_interval <= cfg.train.num_epochs, (
        f"Validation interval is not between 0 and {cfg.train.num_epochs}: {cfg.train.validation_interval}"
    )
    assert 0 <= cfg.train.moving_average_decay < 1, \
        "Moving average decay must be within [0, 1)."
    assert 0 < cfg.train.lr_schedule.warmup_init < 1, \
        "The initial learning rate during warmup must be within (0, 1)."
    assert cfg.train.lr_schedule.learning_rate > 0, \
        "learning_rate must be positive."
    assert cfg.train.num_epochs >= cfg.train.lr_schedule.warmup_epoch >= 0, \
        "warmup_epoch must be within [0, num_epochs]."

    # model config
    assert 'efficientdet-d' in str(cfg.model.name), \
        "model name can be chosen from efficientdet-d0 to efficientdet-d5."
    assert cfg.model.min_level == 3, "min_level must be 3"
    assert cfg.model.max_level == 7, "max_level must be 7"

    # eval config
    assert cfg.evaluate.num_samples >= cfg.evaluate.batch_size > 0, \
        "batch size for evaluation must be (0, num_samples]"
    assert cfg.train.num_epochs >= cfg.evaluate.start_eval_epoch >= 0, \
        "start_eval_epoch must be within [0, num_epochs]."

    # dataset config
    assert cfg.dataset.train_tfrecords, \
        "train_tfrecords must be specified."
    assert cfg.dataset.val_tfrecords, \
        "val_tfrecords must be specified."
    assert 1 < cfg.dataset.num_classes, \
        "num_classes is number of categories + 1 (background). It must be greater than 1."

    # augmentation config
    assert cfg.dataset.augmentation.random_crop_max_scale >= cfg.dataset.augmentation.random_crop_min_scale > 0, \
        "random_crop_min_scale should be positive and no greater than random_crop_max_scale."

    assert cfg.prune.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert cfg.prune.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

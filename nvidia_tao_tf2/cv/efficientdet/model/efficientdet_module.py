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
import re

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase
from tensorflow_quantization.quantize import quantize_model

from nvidia_tao_tf2.blocks.module import TAOModule
from nvidia_tao_tf2.cv.efficientdet.losses import losses
from nvidia_tao_tf2.cv.efficientdet.model.efficientdet import efficientdet
from nvidia_tao_tf2.cv.efficientdet.model import optimizer_builder
from nvidia_tao_tf2.cv.efficientdet.utils import keras_utils
from nvidia_tao_tf2.cv.efficientdet.utils.helper import decode_eff, dump_json, load_model, load_json_model
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import is_main_process, get_world_size
logger = logging.getLogger(__name__)


class EfficientDetModule(TAOModule):
    """EfficientDet Module."""

    def __init__(self, hparams):
        """Init."""
        self.hparams = hparams
        self.steps_per_epoch = (
            hparams.num_examples_per_epoch +
            (hparams.train_batch_size * get_world_size()) - 1) // \
            (hparams.train_batch_size * get_world_size())

        num_samples = (hparams.eval_samples + get_world_size() - 1) // get_world_size()
        self.num_samples = (num_samples + hparams.eval_batch_size - 1) // hparams.eval_batch_size
        self.hparams.eval_samples = self.num_samples

        self.resume_ckpt_path = None
        if hparams.resume_training_checkpoint_path:
            self.initial_epoch = int(hparams.resume_training_checkpoint_path[:-4].split('_')[-1])
            self.resume_ckpt_path = hparams.resume_training_checkpoint_path
        else:
            self.initial_epoch, self.resume_ckpt_path = self._get_latest_checkpoint(hparams.results_dir)

        self.model, self.eval_model = self._build_models(hparams)
        self._load_pretrained_weights(hparams)
        self.configure_optimizers(hparams, self.steps_per_epoch)
        self.configure_losses(hparams)
        if hparams.qat:
            logger.info("QAT enabled.")
            self._quantize_models()
        if is_main_process():
            self.model.summary()
        self.compile()
        self.initial_epoch = self._resume(hparams, self.steps_per_epoch)

    def _quantize_models(self):
        """Quantize models."""
        self.model = quantize_model(self.model, custom_qdq_cases=[EfficientNetQDQCase()])
        self.eval_model = quantize_model(self.eval_model, custom_qdq_cases=[EfficientNetQDQCase()])

    def _build_models(self, hparams):
        """Build train/eval unpruned/pruned models."""
        if not hparams.pruned_model_path:
            if is_main_process():
                logger.info("Building unpruned graph...")
            input_shape = list(hparams.image_size) + [3] \
                if hparams.data_format == 'channels_last' else [3] + list(hparams.image_size)
            original_learning_phase = tf.keras.backend.learning_phase()
            model = efficientdet(input_shape, training=True, config=hparams)
            tf.keras.backend.set_learning_phase(0)
            eval_model = efficientdet(input_shape, training=False, config=hparams)
            tf.keras.backend.set_learning_phase(original_learning_phase)
        else:
            if is_main_process():
                logger.info("Loading pruned graph...")
            original_learning_phase = tf.keras.backend.learning_phase()
            model = load_model(hparams.pruned_model_path, hparams, mode='train')
            tf.keras.backend.set_learning_phase(0)
            eval_model = load_model(hparams.pruned_model_path, hparams, mode='eval')
            tf.keras.backend.set_learning_phase(original_learning_phase)

        # save nonQAT nonAMP graph in results_dir
        dump_json(model, os.path.join(hparams.results_dir, 'train_graph.json'))
        dump_json(eval_model, os.path.join(hparams.results_dir, 'eval_graph.json'))
        return model, eval_model

    def _resume(self, hparams, steps_per_epoch):
        """Resume from checkpoint."""
        if self.resume_ckpt_path:
            print(self.resume_ckpt_path)
            ckpt_path, _ = decode_eff(self.resume_ckpt_path, hparams.encryption_key)
            train_from_epoch = keras_utils.restore_ckpt(
                self.model,
                ckpt_path,
                hparams.moving_average_decay,
                steps_per_epoch=steps_per_epoch,
                expect_partial=True)
            return train_from_epoch
        return self.hparams.init_epoch

    def _load_pretrained_weights(self, hparams):
        """Load pretrained weights."""
        if is_main_process() and not self.resume_ckpt_path:
            if str(hparams.checkpoint).endswith(".tlt"):
                ckpt_path, ckpt_name = decode_eff(
                    str(hparams.checkpoint), hparams.encryption_key)
            else:
                ckpt_path = hparams.checkpoint
            if ckpt_path:
                if not hparams.pruned_model_path:
                    logger.info("Loading pretrained weight...")
                    if 'train_graph.json' in os.listdir(ckpt_path):
                        logger.info("Loading EfficientDet model...")
                        pretrained_model = load_json_model(
                            os.path.join(ckpt_path, 'train_graph.json'))
                        keras_utils.restore_ckpt(
                            pretrained_model,
                            os.path.join(ckpt_path, ckpt_name),
                            hparams.moving_average_decay,
                            steps_per_epoch=0,
                            expect_partial=True)
                    else:
                        logger.info("Loading EfficientNet backbone...")
                        pretrained_model = tf.keras.models.load_model(ckpt_path)
                    for layer in pretrained_model.layers[1:]:
                        # The layer must match up to prediction layers.
                        l_return = None
                        try:
                            l_return = self.model.get_layer(layer.name)
                        except ValueError:
                            # Some layers are not there
                            logger.info("Skipping %s, as it does not exist in the training model.", layer.name)
                        if l_return is not None:
                            try:
                                l_return.set_weights(layer.get_weights())
                            except ValueError:
                                logger.info("Skipping %s, due to shape mismatch.", layer.name)

    def configure_losses(self, hparams, loss=None):
        """Configure losses."""
        loss = loss or {}
        focal_loss = losses.StableFocalLoss(
            hparams.alpha,
            hparams.gamma,
            label_smoothing=hparams.label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE)
        box_loss = losses.BoxLoss(
            hparams.delta,  # TODO(@yuw): add to default config
            reduction=tf.keras.losses.Reduction.NONE)
        box_iou_loss = losses.BoxIouLoss(
            hparams.iou_loss_type,
            hparams.min_level,
            hparams.max_level,
            hparams.num_scales,
            hparams.aspect_ratios,
            hparams.anchor_scale,
            hparams.image_size,
            reduction=tf.keras.losses.Reduction.NONE)
        self.losses = {
            'box_loss': box_loss,
            'box_iou_loss': box_iou_loss,
            'class_loss': focal_loss,
            **loss
        }

    def configure_optimizers(self, hparams, steps_per_epoch):
        """Configure optimizers."""
        self.optimizers = optimizer_builder.get_optimizer(
            hparams, steps_per_epoch)

    def compile(self):
        """Compile model."""
        self.model.compile(
            optimizer=self.optimizers,
            loss=self.losses)
        self.model.train_step = self.train_step
        self.model.test_step = self.test_step

    def train_step(self, data):
        """Train step.

        Args:
        data: Tuple of (images, labels). Image tensor with shape [batch_size,
            height, width, 3]. The height and width are fixed and equal.Input labels
            in a dictionary. The labels include class targets and box targets which
            are dense label maps. The labels are generated from get_input_fn
            function in data/dataloader.py.

        Returns:
        A dict record loss info.
        """
        images, labels = data
        with tf.GradientTape() as tape:
            if len(self.hparams.heads) == 2:
                cls_outputs, box_outputs, seg_outputs = self.model(images, training=True)
            elif 'object_detection' in self.hparams.heads:
                cls_outputs, box_outputs = self.model(images, training=True)
            elif 'segmentation' in self.hparams.heads:
                seg_outputs, = self.model(images, training=True)
                raise ValueError("`segmentation` head is disabled. Please set `object_detection`.")
            total_loss = 0
            loss_vals = {}
            if 'object_detection' in self.hparams.heads:
                det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                                loss_vals)
                total_loss += det_loss
            if 'segmentation' in self.hparams.heads:
                seg_loss_layer = self.model.loss['seg_loss']
                seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
                total_loss += seg_loss
                loss_vals['seg_loss'] = seg_loss

            reg_l2_loss = self._reg_l2_loss(self.hparams.l2_weight_decay) if self.hparams.l2_weight_decay else 0
            reg_l1_loss = self._reg_l1_loss(self.hparams.l1_weight_decay) if self.hparams.l1_weight_decay else 0
            loss_vals['reg_l2_loss'] = reg_l2_loss
            loss_vals['reg_l1_loss'] = reg_l1_loss
            total_loss += (reg_l2_loss + reg_l1_loss)
            if isinstance(self.model.optimizer,
                          tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.model.optimizer.get_scaled_loss(total_loss)
                optimizer = self.model.optimizer._optimizer
            else:
                scaled_loss = total_loss
                optimizer = self.model.optimizer
        compress = keras_utils.get_mixed_precision_policy().compute_dtype == 'float16'
        tape = hvd.DistributedGradientTape(
            tape, compression=hvd.Compression.fp16 if compress else hvd.Compression.none)

        loss_vals['loss'] = total_loss
        loss_vals['learning_rate'] = optimizer.learning_rate(optimizer.iterations)
        trainable_vars = self._freeze_vars()
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        if isinstance(self.model.optimizer,
                      tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.model.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        if self.hparams.clip_gradients_norm > 0:
            clip_norm = abs(self.hparams.clip_gradients_norm)
            gradients = [
                tf.clip_by_norm(g, clip_norm) if g is not None else None
                for g in gradients
            ]
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
            loss_vals['gradient_norm'] = tf.linalg.global_norm(gradients)

        # TODO(@yuw)
        # grads_and_vars = []
        # # Special treatment for biases (beta is named as bias in reference model)
        # for grad, var in zip(gradients, trainable_vars):
        #     if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
        #         grad = 2.0 * grad
        #     grads_and_vars.append((grad, var))
        # self.model.optimizer.apply_gradients(grads_and_vars)

        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss_vals

    def test_step(self, data):
        """Test step.

        Args:
        data: Tuple of (images, labels). Image tensor with shape [batch_size,
            height, width, 3]. The height and width are fixed and equal.Input labels
            in a dictionary. The labels include class targets and box targets which
            are dense label maps. The labels are generated from get_input_fn
            function in data/dataloader.py.

        Returns:
        A dict record loss info.
        """
        images, labels = data
        if len(self.hparams.heads) == 2:
            cls_outputs, box_outputs, seg_outputs = self.model(images, training=False)
        elif 'object_detection' in self.hparams.heads:
            cls_outputs, box_outputs = self.model(images, training=False)
        elif 'segmentation' in self.hparams.heads:
            seg_outputs, = self.model(images, training=False)
        reg_l2loss = self._reg_l2_loss(self.hparams.l2_weight_decay) if self.hparams.l2_weight_decay else 0
        reg_l1loss = self._reg_l1_loss(self.hparams.l1_weight_decay) if self.hparams.l1_weight_decay else 0
        total_loss = reg_l2loss + reg_l1loss
        loss_vals = {}
        if 'object_detection' in self.hparams.heads:
            det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                            loss_vals)
            total_loss += det_loss
        if 'segmentation' in self.hparams.heads:
            seg_loss_layer = self.model.loss['seg_loss']
            seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
            total_loss += seg_loss
            loss_vals['seg_loss'] = seg_loss
        loss_vals['loss'] = total_loss
        return loss_vals

    def _freeze_vars(self):
        """Get trainable variables."""
        if self.hparams.var_freeze_expr:
            return [
                v for v in self.model.trainable_variables
                if not re.match(self.hparams.var_freeze_expr, v.name)
            ]
        return self.model.trainable_variables

    def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
        """Return regularization l2 loss loss."""
        var_match = re.compile(regex)
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self._freeze_vars()
            if var_match.match(v.name)
        ])

    def _reg_l1_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
        """Return regularization l1 loss loss."""
        var_match = re.compile(regex)
        return tf.contrib.layers.apply_regularization(
            tf.keras.regularizers.l1(weight_decay / 2.0),
            [v for v in self._freeze_vars()
             if var_match.match(v.name)])

    def _detection_loss(self, cls_outputs, box_outputs, labels, loss_vals):
        """Computes total detection loss.

        Computes total detection loss including box and class loss from all levels.
        Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width,
            num_anchors * 4].
        labels: the dictionary that returned from dataloader that includes
            groundtruth targets.
        loss_vals: A dict of loss values.

        Returns:
        total_loss: an integer tensor representing total loss reducing from
            class and box losses from all levels.
        cls_loss: an integer tensor representing total class loss.
        box_loss: an integer tensor representing total box regression loss.
        box_iou_loss: an integer tensor representing total box iou loss.
        """
        # convert to float32 for loss computing.
        cls_outputs = [tf.cast(i, tf.float32) for i in cls_outputs]
        box_outputs = [tf.cast(i, tf.float32) for i in box_outputs]

        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
        levels = range(len(cls_outputs))
        cls_losses = []
        box_losses = []

        for level in levels:
            # Onehot encoding for classification labels.
            cls_targets_at_level = tf.one_hot(
                labels[f'cls_targets_{level + 3}'],
                self.hparams.num_classes)

            if self.hparams.data_format == 'channels_first':
                targets_shape = tf.shape(cls_targets_at_level)
                bs = targets_shape[0]
                width = targets_shape[2]
                height = targets_shape[3]
                cls_targets_at_level = tf.reshape(
                    cls_targets_at_level,
                    [bs, -1, width, height])
            else:
                targets_shape = tf.shape(cls_targets_at_level)
                bs = targets_shape[0]
                width = targets_shape[1]
                height = targets_shape[2]
                cls_targets_at_level = tf.reshape(
                    cls_targets_at_level,
                    [bs, width, height, -1])
            box_targets_at_level = labels[f'box_targets_{level + 3}']

            class_loss_layer = self.model.loss.get('class_loss', None)
            if class_loss_layer:
                cls_loss = class_loss_layer([num_positives_sum, cls_targets_at_level],
                                            cls_outputs[level])

                if self.hparams.data_format == 'channels_first':
                    cls_loss = tf.reshape(
                        cls_loss, [bs, -1, width, height, self.hparams.num_classes])
                else:
                    cls_loss = tf.reshape(
                        cls_loss, [bs, width, height, -1, self.hparams.num_classes])
                cls_loss *= tf.cast(
                    tf.expand_dims(
                        tf.not_equal(labels[f'cls_targets_{level + 3}'], -2), -1),
                    tf.float32)
                cls_losses.append(tf.reduce_sum(cls_loss))

            if self.hparams.box_loss_weight and self.model.loss.get('box_loss', None):
                box_loss_layer = self.model.loss['box_loss']
                box_losses.append(
                    box_loss_layer(
                        [num_positives_sum, box_targets_at_level],
                        box_outputs[level]))

        if self.hparams.iou_loss_type:
            box_outputs = tf.concat([tf.reshape(v, [-1, 4]) for v in box_outputs], axis=0)  # noqa pylint: disable=E1123
            box_targets = tf.concat([  # noqa pylint: disable=E1123
                tf.reshape(labels[f'box_targets_{level + 3}'], [-1, 4])
                for level in levels], axis=0)
            box_iou_loss_layer = self.model.loss['box_iou_loss']
            box_iou_loss = box_iou_loss_layer([num_positives_sum, box_targets],
                                              box_outputs)
            loss_vals['box_iou_loss'] = box_iou_loss
        else:
            box_iou_loss = 0

        cls_loss = tf.add_n(cls_losses) if cls_losses else 0
        box_loss = tf.add_n(box_losses) if box_losses else 0
        total_loss = (
            cls_loss + self.hparams.box_loss_weight * box_loss +
            self.hparams.iou_loss_weight * box_iou_loss)
        loss_vals['det_loss'] = total_loss
        loss_vals['cls_loss'] = cls_loss
        loss_vals['box_loss'] = box_loss
        return total_loss

    def _get_latest_checkpoint(self, model_dir, model_name='efficientdet-d'):
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
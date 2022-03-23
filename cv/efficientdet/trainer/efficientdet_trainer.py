from blocks.trainer import Trainer
import re
import tensorflow as tf
import horovod.tensorflow as hvd


class EfficientDetTrainer(Trainer):
    def __init__(self, model, config=None, callbacks=None):
        self.model = model
        self.callbacks = callbacks
        self.config=config

    def fit(self, train_dataset, eval_dataset,
            num_epochs,
            steps_per_epoch,
            initial_epoch,
            validation_steps,
            verbose) -> None:
        self.model.train_step = self.train_step
        self.model.test_step = self.test_step
        history = self.model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=initial_epoch,
            callbacks=self.callbacks,
            verbose=verbose,
            validation_data=eval_dataset,
            validation_steps=validation_steps)
        print(history)

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
            if len(self.config.heads) == 2:
                cls_outputs, box_outputs, seg_outputs = self.model(images, training=True)
            elif 'object_detection' in self.config.heads:
                cls_outputs, box_outputs = self.model(images, training=True)
            elif 'segmentation' in self.config.heads:
                seg_outputs, = self.model(images, training=True)
            total_loss = 0
            loss_vals = {}
            if 'object_detection' in self.config.heads:
                det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                                loss_vals)
                total_loss += det_loss
            if 'segmentation' in self.config.heads:
                seg_loss_layer = self.model.loss['seg_loss']
                seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
                total_loss += seg_loss
                loss_vals['seg_loss'] = seg_loss

            reg_l2_loss = self._reg_l2_loss(self.config.l2_weight_decay) if self.config.l2_weight_decay else 0
            reg_l1_loss = self._reg_l1_loss(self.config.l1_weight_decay) if self.config.l1_weight_decay else 0
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
        tape = hvd.DistributedGradientTape(tape)
        loss_vals['loss'] = total_loss
        loss_vals['learning_rate'] = optimizer.learning_rate(optimizer.iterations)
        trainable_vars = self._freeze_vars()
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        if isinstance(self.model.optimizer,
                      tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.model.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        if self.config.clip_gradients_norm > 0:
            clip_norm = abs(self.config.clip_gradients_norm)
            gradients = [
                tf.clip_by_norm(g, clip_norm) if g is not None else None
                for g in gradients
            ]
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
            loss_vals['gradient_norm'] = tf.linalg.global_norm(gradients)

        # TODO(@yuw): experimental!
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
        if len(self.config.heads) == 2:
            cls_outputs, box_outputs, seg_outputs = self.model(images, training=False)
        elif 'object_detection' in self.config.heads:
            cls_outputs, box_outputs = self.model(images, training=False)
        elif 'segmentation' in self.config.heads:
            seg_outputs, = self.model(images, training=False)
        reg_l2loss = self._reg_l2_loss(self.config.l2_weight_decay) if self.config.l2_weight_decay else 0
        reg_l1loss = self._reg_l1_loss(self.config.l1_weight_decay) if self.config.l1_weight_decay else 0
        total_loss = reg_l2loss + reg_l1loss
        loss_vals = {}
        if 'object_detection' in self.config.heads:
            det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                            loss_vals)
            total_loss += det_loss
        if 'segmentation' in self.config.heads:
            seg_loss_layer = self.model.loss['seg_loss']
            seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
            total_loss += seg_loss
            loss_vals['seg_loss'] = seg_loss
        loss_vals['loss'] = total_loss
        return loss_vals


    def _freeze_vars(self):
        if self.config.var_freeze_expr:
            return [
            v for v in self.model.trainable_variables
            if not re.match(self.config.var_freeze_expr, v.name)
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
            cls_targets_at_level = tf.one_hot(labels['cls_targets_%d' % (level + 3)],
                                            self.config.num_classes)

            if self.config.data_format == 'channels_first':
                targets_shape = tf.shape(cls_targets_at_level)
                bs = targets_shape[0]
                width = targets_shape[2]
                height = targets_shape[3]
                cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                                [bs, -1, width, height])
            else:
                targets_shape = tf.shape(cls_targets_at_level)
                bs = targets_shape[0]
                width = targets_shape[1]
                height = targets_shape[2]
                cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                                [bs, width, height, -1])
            box_targets_at_level = labels['box_targets_%d' % (level + 3)]

            class_loss_layer = self.model.loss.get('class_loss', None)
            if class_loss_layer:
                cls_loss = class_loss_layer([num_positives_sum, cls_targets_at_level],
                                            cls_outputs[level])

                if self.config.data_format == 'channels_first':
                    cls_loss = tf.reshape(
                        cls_loss, [bs, -1, width, height, self.config.num_classes])
                else:
                    cls_loss = tf.reshape(
                        cls_loss, [bs, width, height, -1, self.config.num_classes])
                cls_loss *= tf.cast(
                    tf.expand_dims(
                        tf.not_equal(labels['cls_targets_%d' % (level + 3)], -2), -1),
                    tf.float32)
                cls_losses.append(tf.reduce_sum(cls_loss))

            if self.config.box_loss_weight and self.model.loss.get('box_loss', None):
                box_loss_layer = self.model.loss['box_loss']
                box_losses.append(
                    box_loss_layer([num_positives_sum, box_targets_at_level],
                                box_outputs[level]))

        if self.config.iou_loss_type:
            box_outputs = tf.concat([tf.reshape(v, [-1, 4]) for v in box_outputs],
                                    axis=0)
            box_targets = tf.concat([
                tf.reshape(labels['box_targets_%d' % (level + 3)], [-1, 4])
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
            cls_loss + self.config.box_loss_weight * box_loss +
            self.config.iou_loss_weight * box_iou_loss)
        loss_vals['det_loss'] = total_loss
        loss_vals['cls_loss'] = cls_loss
        loss_vals['box_loss'] = box_loss
        return total_loss

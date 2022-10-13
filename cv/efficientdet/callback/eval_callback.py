# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Callback related utils."""

import os
from mpi4py import MPI
import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage

import common.logging.logging as status_logging

from cv.efficientdet.processor.postprocessor import EfficientDetPostprocessor
from cv.efficientdet.utils import coco_metric
from cv.efficientdet.utils import label_utils
from cv.efficientdet.utils.helper import fetch_optimizer
from cv.efficientdet.utils.horovod_utils import is_main_process
from cv.efficientdet.visualize import vis_utils


class COCOEvalCallback(tf.keras.callbacks.Callback):
    """COCO Evaluation Callback."""

    def __init__(self, eval_dataset, eval_model, eval_freq, start_eval_epoch, eval_params, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.dataset = eval_dataset
        self.eval_model = eval_model
        self.eval_freq = eval_freq
        self.start_eval_epoch = start_eval_epoch
        self.eval_params = eval_params
        self.ema_opt = None
        self.postpc = EfficientDetPostprocessor(self.eval_params)
        log_dir = os.path.join(eval_params.results_dir, 'eval')
        self.file_writer = tf.summary.create_file_writer(log_dir)
        label_map = label_utils.get_label_map(eval_params.evaluate.label_map)
        self.evaluator = coco_metric.EvaluationMetric(
            filename=eval_params.data.val_json_file, label_map=label_map)
        self.pbar = tf.keras.utils.Progbar(eval_params.evaluate.num_samples)
        self.s_logger = status_logging.get_status_logger()

    def set_model(self, model):
        """Set model."""
        if self.eval_params.train.moving_average_decay > 0:
            self.ema_opt = fetch_optimizer(model, MovingAverage)
        return super().set_model(model)

    @tf.function
    def eval_model_fn(self, images, labels):
        """Evaluation model function."""
        cls_outputs, box_outputs = self.eval_model(images, training=False)
        detections = self.postpc.generate_detections(
            cls_outputs, box_outputs,
            labels['image_scales'],
            labels['source_ids'])

        def transform_detections(detections):
            # A transforms detections in [id, x1, y1, x2, y2, score, class]
            # form to [id, x, y, w, h, score, class]."""
            return tf.stack([
                detections[:, :, 0],
                detections[:, :, 1],
                detections[:, :, 2],
                detections[:, :, 3] - detections[:, :, 1],
                detections[:, :, 4] - detections[:, :, 2],
                detections[:, :, 5],
                detections[:, :, 6],
            ], axis=-1)

        tf.numpy_function(
            self.evaluator.update_state,
            [labels['groundtruth_data'], transform_detections(detections)], [])
        return detections, labels['image_scales']

    def evaluate(self, epoch):
        """Run evalution at Nth epoch."""
        if self.eval_params.train.moving_average_decay > 0:
            self.ema_opt.swap_weights()  # get ema weights
        self.eval_model.set_weights(self.model.get_weights())
        self.evaluator.reset_states()
        # evaluate all images.
        for i, (images, labels) in enumerate(self.dataset):
            detections, scales = self.eval_model_fn(images, labels)
            # [id, x1, y1, x2, y2, score, class]
            if self.eval_params.train.image_preview and i == 0:
                bs_index = 0
                image = np.copy(images[bs_index])
                if self.eval_params.data_format == 'channels_first':
                    image = np.transpose(image, (1, 2, 0))
                # decode image
                image = vis_utils.denormalize_image(image)
                predictions = np.array(detections[bs_index])
                predictions[:, 1:5] /= scales[bs_index]
                boxes = predictions[:, 1:5].astype(np.int32)
                boxes = boxes[:, [1, 0, 3, 2]]
                classes = predictions[:, -1].astype(np.int32)
                scores = predictions[:, -2]

                image = vis_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    boxes,
                    classes,
                    scores,
                    {},
                    min_score_thresh=0.3,
                    max_boxes_to_draw=100,
                    line_thickness=2)
                with self.file_writer.as_default():
                    tf.summary.image('Image Preview', tf.expand_dims(image, axis=0), step=epoch)
            # draw detections
            if is_main_process():
                self.pbar.update(i)

        # gather detections from all ranks
        self.evaluator.gather()

        # compute the final eval results.
        if is_main_process():
            metrics = self.evaluator.result()
            metric_dict = {}
            with self.file_writer.as_default(), tf.summary.record_if(True):
                for i, name in enumerate(self.evaluator.metric_names):
                    tf.summary.scalar(name, metrics[i], step=epoch)
                    metric_dict[name] = metrics[i]

            for k, v in metric_dict.items():
                self.s_logger.kpi[k] = float(v)
            try:
                self.s_logger.write(
                    status_level=status_logging.Status.RUNNING)
            except IOError:
                # We let this pass because we do not want the json file writing to crash the whole job.
                pass


        if self.eval_params.train.moving_average_decay > 0:
            self.ema_opt.swap_weights()  # get base weights

        MPI.COMM_WORLD.Barrier()   # noqa pylint: disable=I1101

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end with eval_freq."""
        if (epoch + 1) >= self.start_eval_epoch and (epoch + 1) % self.eval_freq == 0:
            self.evaluate(epoch)

"""Callback related utils."""
from concurrent import futures
import os
from mpi4py import MPI
import numpy as np
import time
import tensorflow as tf
import horovod.tensorflow.keras.callbacks as hvd_callbacks
from tensorflow_addons.optimizers import MovingAverage
from typing import Any, List, MutableMapping, Text

from cv.efficientdet.processor.postprocessor import EfficientDetPostprocessor
from cv.efficientdet.utils import coco_metric
from cv.efficientdet.utils import label_utils
from cv.efficientdet.utils.horovod_utils import get_world_size, is_main_process
from cv.efficientdet.visualize import vis_utils


class BatchTimestamp(object):
    """A structure to store batch time stamp."""

    def __init__(self, batch_index, timestamp):
        self.batch_index = batch_index
        self.timestamp = timestamp

    def __repr__(self):
        return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
                self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, batch_size, logger, log_steps=1, logdir=None):
        """Callback for logging performance.

        Args:
            batch_size: Total batch size.
            log_steps: Interval of steps between logging of batch level stats.
            logdir: Optional directory to write TensorBoard summaries.
        """
        # TODO(wcromar): remove this parameter and rely on `logs` parameter of
        # on_train_batch_end()
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.log_steps = log_steps
        self.last_log_step = 0
        self.steps_before_epoch = 0
        self.steps_in_epoch = 0
        self.start_time = None
        self.logger = logger
        self.step_per_epoch = 0

        if logdir:
            self.summary_writer = tf.summary.create_file_writer(logdir)
        else:
            self.summary_writer = None

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log = []

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log = []
        self.latency = []
        self.throughput = []

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    @property
    def average_steps_per_second(self):
        """The average training steps per second across all epochs."""
        return (self.global_steps - self.step_per_epoch) / sum(self.epoch_runtime_log[1:])

    @property
    def average_examples_per_second(self):
        """The average number of training examples per second across all epochs."""
        # return self.average_steps_per_second * self.batch_size
        ind = int(0.1*len(self.throughput))
        return sum(self.throughput[ind:])/(len(self.throughput[ind:]))

    @property
    def average_time_per_iteration(self):
        """The average time per iteration in seconds across all epochs."""
        ind = int(0.1*len(self.latency))
        return sum(self.latency[ind:])/(len(self.latency[ind:]))

    def on_train_end(self, logs=None):
        self.train_finish_time = time.time()

        if self.summary_writer:
            self.summary_writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

        # Record the timestamp of the first global step
        if not self.timestamp_log:
            self.timestamp_log.append(BatchTimestamp(self.global_steps, self.start_time))

    def on_batch_end(self, batch, logs=None):
        """Records elapse time of the batch and calculates examples per second."""
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size

            self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
            elapsed_time_str='{:.2f} seconds'.format(elapsed_time)
            self.logger.log(
                step='PARAMETER',
                data={
                    'Latency': elapsed_time_str,
                    'fps': examples_per_second,
                    'steps': (self.last_log_step, self.global_steps)})

            if self.summary_writer:
                with self.summary_writer.as_default():
                    tf.summary.scalar('global_step/sec', steps_per_second,
                                                        self.global_steps)
                    tf.summary.scalar('examples/sec', examples_per_second,
                                                        self.global_steps)

            self.last_log_step = self.global_steps
            self.start_time = None
            self.latency.append(elapsed_time)
            self.throughput.append(examples_per_second)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.step_per_epoch = self.steps_in_epoch
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)

        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0


class LRTensorBoard(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.steps_before_epoch = 0
        self.steps_in_epoch = 0

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1

        lr = self.model.optimizer.lr(self.global_steps)
        with self.summary_writer.as_default():
            summary = tf.summary.scalar('learning_rate', lr, self.global_steps)

    def on_epoch_end(self, epoch, logs=None):
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

    def on_train_end(self, logs=None):
        self.summary_writer.flush()


class LoggingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("Iter: {}".format(batch))
        for var in self.model.variables:
            # if 'dense' in var.name:
            #   continue
            print("Var: {} {}".format(var.name, var.value))
            try:
                slot = self.model.optimizer.get_slot(var, "average")
                print("Avg: {}".format(slot))
            except KeyError as e:
                print("{} does not have ema average slot".format(var.name))


def fetch_optimizer(model,opt_type) -> tf.keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""
    
    # this is the case where our target optimizer is not wrapped by any other optimizer(s)
    if isinstance(model.optimizer,opt_type):
        return model.optimizer
    
    # Dive into nested optimizer object until we reach the target opt
    opt = model.optimizer
    while hasattr(opt, '_optimizer'):
        opt = opt._optimizer
        if isinstance(opt,opt_type):
            return opt 
    raise TypeError(f'Failed to find {opt_type} in the nested optimizer object')


class MovingAverageCallback(tf.keras.callbacks.Callback):
    """A Callback to be used with a `MovingAverage` optimizer.

    Applies moving average weights to the model during validation time to test
    and predict on the averaged weights rather than the current model weights.
    Once training is complete, the model weights will be overwritten with the
    averaged weights (by default).

    Attributes:
        overwrite_weights_on_train_end: Whether to overwrite the current model
            weights with the averaged weights from the moving average optimizer.
        **kwargs: Any additional callback arguments.
    """

    def __init__(self,
                 overwrite_weights_on_train_end: bool = False,
                 **kwargs):
        super(MovingAverageCallback, self).__init__(**kwargs)
        self.overwrite_weights_on_train_end = overwrite_weights_on_train_end
        self.ema_opt = None

    def set_model(self, model: tf.keras.Model):
        super(MovingAverageCallback, self).set_model(model)
        self.ema_opt = fetch_optimizer(model, MovingAverage)
        self.ema_opt.shadow_copy(self.model.weights)

    def on_test_begin(self, logs: MutableMapping[Text, Any] = None):
        self.ema_opt.swap_weights()

    def on_test_end(self, logs: MutableMapping[Text, Any] = None):
        self.ema_opt.swap_weights()

    def on_train_end(self, logs: MutableMapping[Text, Any] = None):
        if self.overwrite_weights_on_train_end:
            self.ema_opt.assign_average_vars(self.model.variables)


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Saves and, optionally, assigns the averaged weights.

    Taken from tfa.callbacks.AverageModelCheckpoint [original class].
    NOTE1: The original class has a type check decorator, which prevents passing non-string save_freq (fix: removed)
    NOTE2: The original class may not properly handle layered (nested) optimizer objects (fix: use fetch_optimizer)

    Attributes:
        update_weights: If True, assign the moving average weights
            to the model, and save them. If False, keep the old
            non-averaged weights, but the saved model uses the
            average weights.
        See `tf.keras.callbacks.ModelCheckpoint` for the other args.
    """

    def __init__(self,
                 update_weights: bool,
                 filepath: str,
                 monitor: str = 'val_loss',
                 verbose: int = 0,
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = 'auto',
                 save_freq: str = 'epoch',
                 **kwargs):

        super().__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            **kwargs)
        self.update_weights = update_weights
        self.ema_opt = None

    def set_model(self, model):
        self.ema_opt = fetch_optimizer(model, MovingAverage)
        return  super().set_model(model)

    def _save_model(self, epoch, batch, logs):
        assert isinstance(self.ema_opt, MovingAverage)

        if self.update_weights:
            self.ema_opt.assign_average_vars(self.model.variables)
            return super()._save_model(epoch, batch, logs)
        else:
            # Note: `model.get_weights()` gives us the weights (non-ref)
            # whereas `model.variables` returns references to the variables.
            non_avg_weights = self.model.get_weights()
            self.ema_opt.assign_average_vars(self.model.variables)
            # result is currently None, since `super._save_model` doesn't
            # return anything, but this may change in the future.
            result = super()._save_model(epoch, batch, logs)
            self.model.set_weights(non_avg_weights)
            return result


class StopEarlyCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs, stop_75, **kwargs):
        super(StopEarlyCallback, self).__init__(**kwargs)
        self.num_epochs = num_epochs
        self.stop_75 = stop_75

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) > (0.75 * self.num_epochs) and self.stop_75) or ((epoch + 1) == 300):
            self.model.stop_training = True


class COCOEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_dataset, eval_freq, start_eval_epoch, eval_params, **kwargs):
        super(COCOEvalCallback, self).__init__(**kwargs)
        self.dataset = eval_dataset
        self.eval_freq = eval_freq
        self.start_eval_epoch = start_eval_epoch
        self.eval_params = eval_params
        self.ema_opt = None
        self.postpc = EfficientDetPostprocessor(self.eval_params)
        log_dir = os.path.join(eval_params['results_dir'], 'eval')
        self.file_writer = tf.summary.create_file_writer(log_dir)
        label_map = label_utils.get_label_map(eval_params['eval_config']['label_map'])
        self.evaluator = coco_metric.EvaluationMetric(
            filename=eval_params['data_config']['validation_json_file'], label_map=label_map)

        self.pbar = tf.keras.utils.Progbar(eval_params['eval_config']['eval_samples'])

    def set_model(self, model):
        self.ema_opt = fetch_optimizer(model, MovingAverage)
        return super().set_model(model)

    @tf.function
    def eval_model_fn(self, images, labels):
        cls_outputs, box_outputs = self.model(images, training=False)
        detections = self.postpc.generate_detections(
            cls_outputs, box_outputs,
            labels['image_scales'],
            labels['source_ids'])

        def transform_detections(detections):
            """A transforms detections in [id, x1, y1, x2, y2, score, class]
               form to [id, x, y, w, h, score, class]."""
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
        if self.eval_params['train_config']['moving_average_decay'] > 0:
            self.ema_opt.swap_weights() # get ema weights

        self.evaluator.reset_states()
        # evaluate all images.
        for i, (images, labels) in enumerate(self.dataset):
            detections, scales = self.eval_model_fn(images, labels)
            # [id, x1, y1, x2, y2, score, class]
            if self.eval_params['train_config']['image_preview'] and i == 0:
                bs_index = 0
                image = np.copy(images[bs_index])
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
                    min_score_thresh=0.2,
                    max_boxes_to_draw=100,
                    line_thickness=2)
                with self.file_writer.as_default():
                    tf.summary.image(f'Image Preview', tf.expand_dims(image, axis=0), step=epoch)
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

            # csv format
            csv_metrics = ['AP','AP50','AP75','APs','APm','APl']
            csv_format = ",".join([str(epoch+1)] + [str(round(metric_dict[key] * 100, 2)) for key in csv_metrics])
            print(metric_dict, "csv format:", csv_format)

        if self.eval_params['train_config']['moving_average_decay'] > 0:
            self.ema_opt.swap_weights() # get base weights
        
        MPI.COMM_WORLD.Barrier()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) >= self.start_eval_epoch and (epoch + 1) % self.eval_freq == 0:
            self.evaluate(epoch)


def get_callbacks(params, mode, eval_dataset, logger, profile=False, 
                  time_history=True, log_steps=1, lr_tb=True, benchmark=False):
    """Get callbacks for given params."""
    callbacks = [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
    if is_main_process():
        # if benchmark == False:
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=params['results_dir'], profile_batch='103,106' if profile else 0, histogram_freq = 1)
        callbacks.append(tb_callback)

        if params['train_config']['moving_average_decay']:
            emackpt_callback = AverageModelCheckpoint(
                filepath=os.path.join(params['results_dir'], 'ema_weights', 'emackpt-{epoch:02d}'),
                update_weights=False,
                amp=params['train_config']['amp'],
                verbose=1,
                save_freq='epoch',
                save_weights_only=True,
                period=params['train_config']['checkpoint_period'])
            callbacks.append(emackpt_callback)

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(params['results_dir'], 'ckpt'),
            verbose=1,
            save_freq='epoch',
            save_weights_only=True,
            period=params['train_config']['checkpoint_period'])
        callbacks.append(ckpt_callback)

        if time_history:
            time_callback = TimeHistory(params['train_config']['train_batch_size'] * get_world_size(),
                logger=logger,
                logdir=params['results_dir'],
                log_steps=log_steps)
            callbacks.append(time_callback)

        # log LR in tensorboard
        if lr_tb == True and benchmark == False:
            callbacks.append(LRTensorBoard(log_dir=params['results_dir']))

    # for large batch sizes training schedule of 350/400 epochs gives better mAP
    # but the best mAP is generally reached fter 75% of the training schedule.
    # So we can stop training at that point or continue to train until 300 epochs
    stop_75 = False if 'eval' in mode else True
    early_stopping = StopEarlyCallback(params['train_config']['num_epochs'], stop_75=stop_75)
    callbacks.append(early_stopping)

    if 'eval' in mode:
        cocoeval = COCOEvalCallback(
            eval_dataset, 
            eval_freq=params['train_config']['checkpoint_period'], 
            start_eval_epoch=1, # TODO
            eval_params=params)
        callbacks.append(cocoeval)

    if params['train_config']['moving_average_decay']:
        callbacks.append(MovingAverageCallback())

    return callbacks

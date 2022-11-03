"""Callback related utils."""

import os
import tensorflow as tf
import horovod.tensorflow.keras.callbacks as hvd_callbacks

from wandb.keras import WandbCallback

from nvidia_tao_tf2.common.mlops.wandb import is_wandb_initialized
from nvidia_tao_tf2.cv.efficientdet.callback.eff_ema_checkpoint import EffEmaCheckpoint
from nvidia_tao_tf2.cv.efficientdet.callback.eff_checkpoint import EffCheckpoint
from nvidia_tao_tf2.cv.efficientdet.callback.eval_callback import COCOEvalCallback
from nvidia_tao_tf2.cv.efficientdet.callback.lr_tensorboard import LRTensorBoard
from nvidia_tao_tf2.cv.efficientdet.callback.logging_callback import MetricLogging
from nvidia_tao_tf2.cv.efficientdet.callback.moving_average_callback import MovingAverageCallback
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import is_main_process


def get_callbacks(params, eval_dataset, steps_per_epoch,
                  eval_model=None, initial_epoch=0):
    """Get callbacks for given params."""
    callbacks = [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
    if is_main_process():
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=params['results_dir'], profile_batch=0, histogram_freq=1)
        callbacks.append(tb_callback)
        # set up checkpointing callbacks
        ckpt_dir = os.path.join(params['results_dir'], 'weights')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        if params['train']['moving_average_decay'] > 0:
            ckpt_callback = EffEmaCheckpoint(
                eff_dir=ckpt_dir,
                key=params['key'],
                update_weights=False,
                amp=params['train']['amp'],
                verbose=0,
                save_freq='epoch',
                save_weights_only=True,
                period=params['train']['checkpoint_interval'],
                is_qat=params['train']['qat'])
        else:
            ckpt_callback = EffCheckpoint(
                eff_dir=ckpt_dir,
                key=params['key'],
                verbose=0,
                save_freq='epoch',
                save_weights_only=True,
                period=params['train']['checkpoint_interval'],
                is_qat=params['train']['qat'])
        callbacks.append(ckpt_callback)

        model_callback = EffCheckpoint(
            eff_dir=params['results_dir'],
            key=params['key'],
            graph_only=True,
            verbose=0,
            save_freq='epoch',
            save_weights_only=True,
            period=params['train']['checkpoint_interval'])
        callbacks.append(model_callback)

        # log LR in tensorboard
        callbacks.append(LRTensorBoard(steps_per_epoch, initial_epoch, log_dir=params['results_dir']))
        # status logging
        callbacks.append(MetricLogging(params['train']['num_epochs'], steps_per_epoch, initial_epoch))

        # Setup the wandb logging callback if weights
        # and biases have been initialized.
        if is_wandb_initialized():
            callbacks.append(WandbCallback())

    cocoeval = COCOEvalCallback(
        eval_dataset,
        eval_model=eval_model,
        eval_freq=params['train']['checkpoint_interval'],
        start_eval_epoch=params['evaluate']['start_eval_epoch'],
        eval_params=params)
    callbacks.append(cocoeval)

    if params['train']['moving_average_decay']:
        callbacks.append(MovingAverageCallback())

    return callbacks

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


def get_callbacks(hparams, eval_dataset, steps_per_epoch,
                  eval_model=None, initial_epoch=0):
    """Get callbacks for given hparams."""
    callbacks = [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
    if is_main_process():
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(hparams['results_dir'], 'tb_events'),
            profile_batch=0, histogram_freq=1)
        callbacks.append(tb_callback)
        # set up checkpointing callbacks
        ckpt_dir = os.path.join(hparams['results_dir'], 'weights')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        if hparams['moving_average_decay'] > 0:
            ckpt_callback = EffEmaCheckpoint(
                eff_dir=ckpt_dir,
                encryption_key=hparams['encryption_key'],
                update_weights=False,
                amp=hparams['mixed_precision'],
                verbose=0,
                save_freq='epoch',
                save_weights_only=True,
                period=hparams['checkpoint_interval'],
                is_qat=hparams['qat'])
        else:
            ckpt_callback = EffCheckpoint(
                eff_dir=ckpt_dir,
                encryption_key=hparams['encryption_key'],
                verbose=0,
                save_freq='epoch',
                save_weights_only=True,
                period=hparams['checkpoint_interval'],
                is_qat=hparams['qat'])
        callbacks.append(ckpt_callback)

        model_callback = EffCheckpoint(
            eff_dir=hparams['results_dir'],
            encryption_key=hparams['encryption_key'],
            graph_only=True,
            verbose=0,
            save_freq='epoch',
            save_weights_only=True,
            period=hparams['checkpoint_interval'])
        callbacks.append(model_callback)

        # log LR in tensorboard
        callbacks.append(
            LRTensorBoard(
                steps_per_epoch,
                initial_epoch,
                log_dir=os.path.join(hparams['results_dir'], 'tb_events')))
        # status logging
        callbacks.append(MetricLogging(hparams['num_epochs'], steps_per_epoch, initial_epoch))

        # Setup the wandb logging callback if weights
        # and biases have been initialized.
        if is_wandb_initialized():
            callbacks.append(WandbCallback())

    cocoeval = COCOEvalCallback(
        eval_dataset,
        eval_model=eval_model,
        eval_freq=hparams['checkpoint_interval'],
        start_eval_epoch=hparams['eval_start'],
        hparams=hparams)
    callbacks.append(cocoeval)

    if hparams['moving_average_decay']:
        callbacks.append(MovingAverageCallback())

    return callbacks

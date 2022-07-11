"""Callback related utils."""
import os
import tensorflow as tf
import horovod.tensorflow.keras.callbacks as hvd_callbacks

from cv.efficientdet.callback.eff_ema_checkpoint import EffEmaCheckpoint
from cv.efficientdet.callback.eff_checkpoint import EffCheckpoint
from cv.efficientdet.callback.eval_callback import COCOEvalCallback
from cv.efficientdet.callback.lr_tensorboard import LRTensorBoard
from cv.efficientdet.callback.moving_average_callback import MovingAverageCallback
from cv.efficientdet.utils.horovod_utils import is_main_process


def get_callbacks(params, eval_dataset, eval_model=None):
    """Get callbacks for given params."""
    callbacks = [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
    if is_main_process():
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=params['train']['results_dir'], profile_batch=0, histogram_freq=1)
        callbacks.append(tb_callback)
        # set up checkpointing callbacks
        ckpt_dir = os.path.join(params['train']['results_dir'], 'weights')
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
                period=params['train']['checkpoint_period'])
        else:
            ckpt_callback = EffCheckpoint(
                eff_dir=ckpt_dir,
                key=params['key'],
                verbose=0,
                save_freq='epoch',
                save_weights_only=True,
                period=params['train']['checkpoint_period'])
        callbacks.append(ckpt_callback)

        model_callback = EffCheckpoint(
            eff_dir=params['train']['results_dir'],
            key=params['key'],
            graph_only=True,
            verbose=0,
            save_freq='epoch',
            save_weights_only=True,
            period=params['train']['checkpoint_period'])
        callbacks.append(model_callback)

        # log LR in tensorboard
        callbacks.append(LRTensorBoard(log_dir=params['train']['results_dir']))

    cocoeval = COCOEvalCallback(
        eval_dataset,
        eval_model=eval_model,
        eval_freq=params['train']['checkpoint_period'],
        start_eval_epoch=1,  # TODO(@yuw): make it configurable
        eval_params=params)
    callbacks.append(cocoeval)

    if params['train']['moving_average_decay']:
        callbacks.append(MovingAverageCallback())

    return callbacks

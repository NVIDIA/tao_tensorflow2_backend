"""Callback related utils."""
import os
import tensorflow as tf
import horovod.tensorflow.keras.callbacks as hvd_callbacks

from cv.efficientdet.callback.average_model_checkpoint import AverageModelCheckpoint
from cv.efficientdet.callback.eval_callback import COCOEvalCallback
# from cv.efficientdet.callback.logging_callback import LoggingCallback
from cv.efficientdet.callback.lr_tensorboard import LRTensorBoard
from cv.efficientdet.callback.moving_average_callback import MovingAverageCallback
from cv.efficientdet.callback.time_history import TimeHistory
from cv.efficientdet.utils.horovod_utils import get_world_size, is_main_process



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

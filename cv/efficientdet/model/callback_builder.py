"""Callback related utils."""
import os
import tensorflow as tf
import horovod.tensorflow.keras.callbacks as hvd_callbacks

# from cv.efficientdet.callback.average_model_checkpoint import AverageModelCheckpoint
from cv.efficientdet.callback.eff_ema_checkpoint import EffEmaCheckpoint
from cv.efficientdet.callback.eff_checkpoint import EffCheckpoint
from cv.efficientdet.callback.eval_callback import COCOEvalCallback
# from cv.efficientdet.callback.logging_callback import LoggingCallback
from cv.efficientdet.callback.lr_tensorboard import LRTensorBoard
from cv.efficientdet.callback.moving_average_callback import MovingAverageCallback
from cv.efficientdet.callback.time_history import TimeHistory
from cv.efficientdet.utils.horovod_utils import get_world_size, is_main_process



def get_callbacks(params, mode, eval_dataset, logger, profile=False, 
                  time_history=True, log_steps=1, lr_tb=True, benchmark=False,
                  eval_model=None):
    """Get callbacks for given params."""
    callbacks = [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
    if is_main_process():
        # if benchmark == False:
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=params['train']['results_dir'], profile_batch='103,106' if profile else 0, histogram_freq = 1)
        callbacks.append(tb_callback)
        # set up checkpointing callbacks
        ckpt_dir = os.path.join(params['train']['results_dir'], 'weights')
        if not os.path.exists(ckpt_dir):
            os.mkdirs(ckpt_dir, exist_ok=True)
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

        if time_history:
            time_callback = TimeHistory(params['train']['batch_size'] * get_world_size(),
                logger=logger,
                logdir=params['train']['results_dir'],
                log_steps=log_steps)
            callbacks.append(time_callback)

        # log LR in tensorboard
        if lr_tb == True and benchmark == False:
            callbacks.append(LRTensorBoard(log_dir=params['train']['results_dir']))


    if 'eval' in mode:
        cocoeval = COCOEvalCallback(
            eval_dataset,
            eval_model=eval_model,
            eval_freq=params['train']['checkpoint_period'], 
            start_eval_epoch=1, # TODO
            eval_params=params)
        callbacks.append(cocoeval)

    if params['train']['moving_average_decay']:
        callbacks.append(MovingAverageCallback())

    return callbacks

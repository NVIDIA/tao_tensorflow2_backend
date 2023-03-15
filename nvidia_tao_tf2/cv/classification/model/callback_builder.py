
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Classification callback builder."""
import os
import horovod.tensorflow.keras as hvd
from wandb.keras import WandbCallback

from tensorflow.keras.callbacks import TensorBoard

from nvidia_tao_tf2.common.mlops.wandb import is_wandb_initialized
from nvidia_tao_tf2.cv.classification.callback.cvs_callback import CSVLoggerWithStatus
from nvidia_tao_tf2.cv.classification.callback.eff_checkpoint import EffCheckpoint
from nvidia_tao_tf2.cv.classification.utils.helper import build_lr_scheduler


def setup_callbacks(ckpt_freq, results_dir, lr_config,
                    init_epoch, iters_per_epoch, max_epoch, key):
    """Setup callbacks: tensorboard, checkpointer, lrscheduler, csvlogger.

    Args:
        ckpt_freq (int): checkpoint and validation frequency.
        results_dir (str): Path to a folder where various training outputs will
                           be written.
        init_epoch (int): The number of epoch to resume training.
        key (str): encryption key
    Returns:
        callbacks (list of keras.callbacks): list of callbacks.
    """
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback()]
    max_iterations = iters_per_epoch * max_epoch
    lrscheduler = build_lr_scheduler(lr_config, hvd.size(), max_iterations)
    init_step = (init_epoch - 1) * iters_per_epoch
    lrscheduler.reset(init_step)
    callbacks.append(lrscheduler)

    if hvd.rank() == 0:
        # Set up the checkpointer.
        save_weights_dir = os.path.join(results_dir, 'weights')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        if not os.path.exists(save_weights_dir):
            os.makedirs(save_weights_dir)
        # Save encrypted models
        checkpointer = EffCheckpoint(save_weights_dir, key, verbose=0, ckpt_freq=ckpt_freq)
        callbacks.append(checkpointer)

        # Set up the custom TensorBoard callback. It will log the loss
        # after every step, and some images and user-set summaries only on
        # the first step of every epoch.
        tensorboard = TensorBoard(log_dir=results_dir)
        callbacks.append(tensorboard)

        # Set up the CSV logger, logging statistics after every epoch.
        csvfilename = os.path.join(results_dir, 'training.csv')
        csvlogger = CSVLoggerWithStatus(
            csvfilename,
            separator=',',
            append=True)
        callbacks.append(csvlogger)
        if is_wandb_initialized():
            callbacks.append(WandbCallback())

    return callbacks

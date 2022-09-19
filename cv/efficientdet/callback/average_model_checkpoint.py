# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Callback related utils."""

import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage

from cv.efficientdet.utils.helper import fetch_optimizer


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
        """Init."""
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
        """Set model."""
        self.ema_opt = fetch_optimizer(model, MovingAverage)
        return super().set_model(model)

    def _save_model(self, epoch, batch, logs):
        """Save model."""
        assert isinstance(self.ema_opt, MovingAverage)

        if self.update_weights:
            self.ema_opt.assign_average_vars(self.model.variables)
            return super()._save_model(epoch, batch, logs)
        # Note: `model.get_weights()` gives us the weights (non-ref)
        # whereas `model.variables` returns references to the variables.
        non_avg_weights = self.model.get_weights()
        self.ema_opt.assign_average_vars(self.model.variables)
        # result is currently None, since `super._save_model` doesn't
        # return anything, but this may change in the future.
        result = super()._save_model(epoch, batch, logs)
        self.model.set_weights(non_avg_weights)
        return result

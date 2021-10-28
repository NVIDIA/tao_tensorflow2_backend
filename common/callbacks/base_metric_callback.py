# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Base metric callback."""

from abc import ABC, abstractmethod
from keras import backend as K
from keras.callbacks import Callback


class BaseMetricCallback(ABC, Callback):
    '''
    Callback function to calculate model metric per k epoch.

    To be implemented in child classes:
    _calc_metric(self, logs): calculate metric and stores to log
    _skip_metric(self, logs): write np.nan (or other values) for metrics to log
    '''

    def __init__(self, eval_model, metric_interval, last_epoch=None, verbose=1):
        '''init function.'''
        metric_interval = int(metric_interval)
        self.metric_interval = metric_interval if metric_interval > 0 else 1
        self.eval_model = eval_model
        self.verbose = verbose
        self.last_epoch = last_epoch

    @abstractmethod
    def _calc_metric(self, logs):
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def _skip_metric(self, logs):
        raise NotImplementedError("Method not implemented!")

    def _get_metric(self, logs):
        K.set_learning_phase(0)
        # First copy weights from training model
        self.eval_model.set_weights(self.model.get_weights())
        self._calc_metric(logs)
        K.set_learning_phase(1)

    def on_epoch_end(self, epoch, logs):
        '''evaluates on epoch end.'''

        if (epoch + 1) % self.metric_interval != 0 and (epoch + 1) != self.last_epoch:
            self._skip_metric(logs)
        else:
            self._get_metric(logs)

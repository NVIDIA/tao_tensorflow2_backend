# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EFF EMA Checkpoint Callback."""

import os
import shutil
import tempfile

import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage

from nvidia_tao_tf2.cv.efficientdet.utils.helper import fetch_optimizer
from nvidia_tao_tf2.cv.efficientdet.utils.helper import dump_json, dump_eval_json, encode_eff


class EffEmaCheckpoint(tf.keras.callbacks.ModelCheckpoint):
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
                 eff_dir: str,
                 encryption_key: str,
                 update_weights: bool,
                 monitor: str = 'val_loss',
                 verbose: int = 0,
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = 'auto',
                 save_freq: str = 'epoch',
                 is_qat: bool = False,
                 **kwargs):
        """Init."""
        super().__init__(
            eff_dir,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            **kwargs)
        self.update_weights = update_weights
        self.ema_opt = None
        self.eff_dir = eff_dir
        self.encryption_key = encryption_key
        self.is_qat = is_qat

    def set_model(self, model):
        """Set model."""
        self.ema_opt = fetch_optimizer(model, MovingAverage)
        return super().set_model(model)

    def _save_model(self, epoch, batch, logs):
        """Save model."""
        assert isinstance(self.ema_opt, MovingAverage), "optimizer must be wrapped in MovingAverage"

        if self.update_weights:
            self.ema_opt.assign_average_vars(self.model.variables)
            super()._save_model(epoch, batch, logs)
        else:
            # Note: `model.get_weights()` gives us the weights (non-ref)
            # whereas `model.variables` returns references to the variables.
            non_avg_weights = self.model.get_weights()
            self.ema_opt.assign_average_vars(self.model.variables)
            # result is currently None, since `super._save_model` doesn't
            # return anything, but this may change in the future.
            super()._save_model(epoch, batch, logs)
            self.model.set_weights(non_avg_weights)

    def _remove_tmp_files(self):
        """Remove temporary zip file and directory."""
        # TODO(@yuw): try catch?
        os.remove(self.temp_zip_file)
        shutil.rmtree(os.path.dirname(self.filepath))

    def on_epoch_end(self, epoch, logs=None):
        """Override on_epoch_end."""
        self.epochs_since_last_save += 1
        eff_epoch = epoch + 1  # eff name started with 001
        checkpoint_dir = tempfile.mkdtemp()
        self.filepath = os.path.join(checkpoint_dir, f'emackpt-{epoch:03d}')  # override filepath

        # pylint: disable=protected-access
        if self.save_freq == 'epoch' and self.epochs_since_last_save >= self.period:
            self._save_model(epoch=epoch, batch=None, logs=logs)  # To self.filepath
            # WORKAROUND to save QAT graph
            if self.is_qat:
                shutil.copy(os.path.join(self.eff_dir, 'train_graph.json'), checkpoint_dir)
                shutil.copy(os.path.join(self.eff_dir, 'eval_graph.json'), checkpoint_dir)
            else:
                # save train/eval graph json to checkpoint_dir
                dump_json(self.model, os.path.join(checkpoint_dir, 'train_graph.json'))
                dump_eval_json(checkpoint_dir, eval_graph='eval_graph.json')
            # convert content in self.filepath to EFF
            eff_filename = f'{self.model.name}_{eff_epoch:03d}.tlt'
            eff_model_path = os.path.join(self.eff_dir, eff_filename)
            self.temp_zip_file = encode_eff(
                checkpoint_dir,
                eff_model_path, self.encryption_key)
            self._remove_tmp_files()

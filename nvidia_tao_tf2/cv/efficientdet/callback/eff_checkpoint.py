# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""EFF Checkpoint Callback."""

import os
import shutil
import tempfile

import tensorflow as tf
from nvidia_tao_tf2.cv.efficientdet.utils.helper import dump_json, dump_eval_json, encode_eff


class EffCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Saves and, optionally, assigns the averaged weights.

    Attributes:
        See `tf.keras.callbacks.ModelCheckpoint` for the other args.
    """

    def __init__(self,
                 eff_dir: str,
                 key: str,
                 graph_only: bool = False,
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
        self.eff_dir = eff_dir
        self.passphrase = key
        self.graph_only = graph_only
        self.is_qat = is_qat

    def _remove_tmp_files(self):
        """Remove temporary zip file and directory."""
        shutil.rmtree(os.path.dirname(self.filepath))
        os.remove(self.temp_zip_file)

    def on_epoch_end(self, epoch, logs=None):
        """Override on_epoch_end."""
        self.epochs_since_last_save += 1
        eff_epoch = epoch + 1  # eff name started with 001
        checkpoint_dir = tempfile.mkdtemp()
        self.filepath = os.path.join(checkpoint_dir, f'ckpt-{epoch:03d}')  # override filepath

        # pylint: disable=protected-access
        if self.save_freq == 'epoch' and self.epochs_since_last_save >= self.period:
            self._save_model(epoch=epoch, batch=None, logs=logs)  # To self.filepath
            if self.graph_only:
                eff_filename = f"{self.model.name}.resume"
            else:
                eff_filename = f'{self.model.name}_{eff_epoch:03d}.tlt'
            # WORKAROUND to save QAT graph
            if self.is_qat:
                shutil.copy(os.path.join(os.path.dirname(self.eff_dir), 'train_graph.json'), checkpoint_dir)
                shutil.copy(os.path.join(os.path.dirname(self.eff_dir), 'eval_graph.json'), checkpoint_dir)
            else:
                # save train/eval graph json to checkpoint_dir
                dump_json(self.model, os.path.join(checkpoint_dir, 'train_graph.json'))
                dump_eval_json(checkpoint_dir, eval_graph='eval_graph.json')
            eff_model_path = os.path.join(self.eff_dir, eff_filename)
            # convert content in self.filepath to EFF
            self.temp_zip_file = encode_eff(
                os.path.dirname(self.filepath),
                eff_model_path, self.passphrase)
            self._remove_tmp_files()

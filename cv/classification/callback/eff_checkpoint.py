# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
"""Callbacks: utilities called at certain points during model training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
from zipfile import ZipFile

from tensorflow.keras.callbacks import ModelCheckpoint

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback


class EffCheckpoint(ModelCheckpoint):
    """Save the encrypted model after every epoch.

    Attributes:
        ENC_KEY: API key to encrypt the model.
        epocs_since_last_save: Number of epochs since model was last saved.
        save_best_only: Flag to save model with best accuracy.
        best: saved instance of best model.
        verbose: Enable verbose messages.
    """

    def __init__(self, eff_dir, key,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 ckpt_freq=1,
                 options=None,
                 **kwargs):
        """Initialization with encryption key."""
        super(EffCheckpoint, self).__init__(
            eff_dir,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options)
        self.passphrase = key
        self.epochs_since_last_save = 0
        self.eff_dir = eff_dir
        self.ckpt_freq = ckpt_freq

    def zipdir(self, src, zip_path) -> None:
        """
        Function creates zip archive from src in dst location. The name of archive is zip_name.
        :param src: Path to directory to be archived.
        :param dst: Path where archived dir will be stored.
        :param zip_name: The name of the archive.
        :return: None
        """
        ### destination directory
        os.chdir(os.path.dirname(zip_path))
        ### zipfile handler
        with ZipFile(zip_path, "w") as zf:
            ### writing content of src directory to the archive
            for root, _, filenames in os.walk(src):
                for filename in filenames:
                    zf.write(
                        os.path.join(root, filename),
                        arcname=os.path.join(root.replace(src, ""),
                        filename))

    def _save_eff(self, epoch, metadata={}):
        """Save EFF Archive."""
        epoch += 1
        os_handle, self.temp_zip_file = tempfile.mkstemp()
        os.close(os_handle)
        # create zipfile from saved_model directory
        self.zipdir(self.filepath, self.temp_zip_file)
        # create artifacts from zipfile
        eff_filename = f'{self.model.name}_{epoch:03d}.tlt'
        zip_art = File(
            name=eff_filename,
            description="Artifact from checkpoint",
            filepath=self.temp_zip_file,
            encryption=True,
            content_callback=BinaryContentCallback,
        )
        eff_filepath = os.path.join(self.eff_dir, eff_filename)
        Archive.save_artifact(
            save_path=eff_filepath, artifact=zip_art, passphrase=self.passphrase, **metadata)

    def _remove_tmp_files(self):
        """Remove temporary zip file and directory."""
        shutil.rmtree(self.filepath)
        os.remove(self.temp_zip_file)

    def on_epoch_end(self, epoch, logs=None):
        """Override on_epoch_end."""
        self.epochs_since_last_save += 1

        # pylint: disable=protected-access
        if self.save_freq == 'epoch' and self.epochs_since_last_save % self.ckpt_freq == 0:
            self.filepath = tempfile.mkdtemp() # override filepath
            self._save_model(epoch=epoch, batch=None, logs=logs)
            self._save_eff(epoch=epoch)
            self._remove_tmp_files()

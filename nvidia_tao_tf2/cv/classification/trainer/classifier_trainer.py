# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Classification Trainer."""
import logging

from nvidia_tao_tf2.blocks.trainer import Trainer
logger = logging.getLogger(__name__)


class ClassifierTrainer(Trainer):
    """Classifier Trainer."""

    def __init__(self, num_epochs, callbacks=None, cfg=None):
        """Init."""
        self.num_epochs = num_epochs
        self.callbacks = callbacks
        self.cfg = cfg

    def fit(self,
            module,
            train_dataset,
            eval_dataset,
            verbose) -> None:
        """Run model.fit with custom steps."""
        if module.initial_epoch < self.num_epochs:
            module.model.fit(
                train_dataset,
                epochs=self.num_epochs,
                steps_per_epoch=module.steps_per_epoch,
                initial_epoch=module.initial_epoch,
                callbacks=self.callbacks,
                verbose=verbose,
                workers=self.cfg['train']['n_workers'],
                validation_data=eval_dataset,
                validation_steps=len(eval_dataset),
                validation_freq=1)
        else:
            logger.info("Training (%d epochs) has finished.", self.num_epochs)

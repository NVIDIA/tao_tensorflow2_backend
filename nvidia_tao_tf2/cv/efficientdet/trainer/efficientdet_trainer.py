# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""EfficientDet Trainer."""
import logging

from nvidia_tao_tf2.blocks.trainer import Trainer
logger = logging.getLogger(__name__)


class EfficientDetTrainer(Trainer):
    """EfficientDet Trainer."""

    def __init__(self, num_epochs, qat=False, callbacks=None):
        """Init."""
        self.num_epochs = num_epochs
        self.callbacks = callbacks
        self.qat = qat

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
                validation_data=eval_dataset,
                validation_steps=module.num_samples)
        else:
            logger.info("Training (%d epochs) has finished.", self.num_epochs)

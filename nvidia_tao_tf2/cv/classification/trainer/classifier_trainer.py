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
            verbose,
            validation_freq: int = 1) -> None:
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
                validation_freq=validation_freq)
        else:
            logger.info("Training (%d epochs) has finished.", self.num_epochs)

# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
"""The main training script."""
import os
import time
from mpi4py import MPI
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger

from efficientdet.config.hydra_runner import hydra_runner
from efficientdet.config.default_config import ExperimentConfig
from efficientdet.dataloader import dataloader
from efficientdet.utils.config_utils import generate_params_from_cfg
from efficientdet.utils import hparams_config
# from model import anchors, callback_builder, coco_metric, dataloader
# from model import efficientdet_keras, label_util, optimizer_builder, postprocess
# from utils import hparams_config, model_utils, setup, train_lib, util_keras
from efficientdet.utils.horovod_utils import is_main_process, get_world_size, get_rank, initialize


def run_experiment(cfg, results_dir, key):

    # get e2e training time
    begin = time.time()
    logging.info("Training started at: {}".format(time.asctime()))

    hvd.init()

    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg['model_config']['model_name'])
    config.update(generate_params_from_cfg(config, cfg, mode='train'))

    # initialize
    initialize(config, training=True)

    steps_per_epoch = (cfg['train_config']['num_examples_per_epoch'] + 
        (cfg['train_config']['train_batch_size'] * get_world_size()) - 1) // \
            (cfg['train_config']['train_batch_size'] * get_world_size())

    # Set up dataloader
    train_dataloader = dataloader.InputReader(
        cfg['data_config']['training_file_pattern'],
        is_training=True,
        use_fake_data=cfg['data_config']['use_fake_data'],
        max_instances_per_image=config.max_instances_per_image)

    eval_dataloader = dataloader.InputReader(
        cfg['data_config']['validation_file_pattern'],
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of MakeNet application.
    """
    run_experiment(cfg=cfg,
                   results_dir=cfg.results_dir,
                   key=cfg.key)


if __name__ == '__main__':
    main()
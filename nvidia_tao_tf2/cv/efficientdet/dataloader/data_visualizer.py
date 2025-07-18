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
"""EfficientDet standalone visualization tool."""
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from PIL import Image

from nvidia_tao_core.config.efficientdet_tf2.default_config import ExperimentConfig

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner

from nvidia_tao_tf2.cv.efficientdet.dataloader import dataloader, datasource
from nvidia_tao_tf2.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.visualize import vis_utils


def visualize(cfg):
    """Run single image visualization."""
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    MODE = 'vis'
    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode=MODE))

    # Set up dataloader
    eval_sources = datasource.DataSource(
        cfg.dataset.val_tfrecords,
        cfg.dataset.val_dirs)
    eval_dl = dataloader.CocoDataset(
        eval_sources,
        is_training=True,
        use_fake_data=False,
        max_instances_per_image=config.max_instances_per_image)
    eval_dataset = eval_dl(
        config.as_dict(),
        batch_size=1)
    iterator = iter(eval_dataset)
    counter = 1
    for next_element in iterator:
        # next_element = iterator.get_next()
        image = next_element[0][0, ...]  # h, w, c
        image = image.numpy()
        image2v = vis_utils.denormalize_image(image)
        Image.fromarray(image2v).save(os.path.join(cfg.results_dir, f'dl_00{counter}.png'))
        counter += 1
        if counter > 10:
            break

    print("Finished visualization.")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="vis", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet evaluation."""
    visualize(cfg)


if __name__ == '__main__':
    main()

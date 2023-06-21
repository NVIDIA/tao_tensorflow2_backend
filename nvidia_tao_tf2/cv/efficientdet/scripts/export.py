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

"""Export EfficientDet model to etlt and TRT engine."""
import logging
import os
import tempfile

import tensorflow as tf
from tensorflow.python.util import deprecation

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf2.common.utils import update_results_dir

from nvidia_tao_tf2.cv.efficientdet.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.efficientdet.exporter.onnx_exporter import EfficientDetGraphSurgeon
from nvidia_tao_tf2.cv.efficientdet.inferencer import inference
from nvidia_tao_tf2.cv.efficientdet.utils import helper, hparams_config
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import initialize
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='efficientdet', mode='export')
def run_export(cfg):
    """Launch EfficientDet export."""
    # Parse and update hparams
    MODE = 'export'
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode=MODE))
    params = config.as_dict()

    assert cfg.export.onnx_file.endswith('.onnx'), "Exported file must end with .onnx"
    output_dir = tempfile.mkdtemp()
    tf.keras.backend.set_learning_phase(0)

    # Load model from graph json
    model = helper.load_model(cfg.export.checkpoint, cfg, MODE, is_qat=cfg.train.qat)
    model.summary()
    # Get input shape from model
    input_shape = list(model.layers[0].input_shape[0])
    max_batch_size = 1 if cfg.export.dynamic_batch_size else cfg.export.batch_size
    input_shape[0] = max_batch_size
    # Build inference model
    export_model = inference.InferenceModel(
        model, config.image_size, params, max_batch_size,
        min_score_thresh=0.001,  # a small value
        max_boxes_to_draw=cfg.evaluate.max_detections_per_image)
    export_model.infer = tf.function(export_model.infer)
    tf.saved_model.save(
        export_model,
        output_dir,
        signatures=export_model.infer.get_concrete_function(
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8)))

    # convert to onnx
    effdet_gs = EfficientDetGraphSurgeon(
        output_dir,
        legacy_plugins=False,
        dynamic_batch=cfg.export.dynamic_batch_size,
        is_qat=cfg.train.qat)
    effdet_gs.update_preprocessor('NHWC', input_shape, preprocessor="imagenet")
    effdet_gs.update_shapes()
    effdet_gs.update_nms(threshold=cfg.export.min_score_thresh)
    # convert onnx to eff
    onnx_file = effdet_gs.save(cfg.export.onnx_file)
    logger.info("The exported model is saved at: %s", onnx_file)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="export", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """Wrapper function for EfficientDet export."""
    cfg = update_results_dir(cfg, 'export')
    initialize(cfg, logger, training=False)
    run_export(cfg=cfg)


if __name__ == '__main__':
    main()

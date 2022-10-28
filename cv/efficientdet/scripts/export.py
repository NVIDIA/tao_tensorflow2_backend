# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Export EfficientDet model to etlt and TRT engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf
from tensorflow.python.util import deprecation

from common.decorators import monitor_status
from common.hydra.hydra_runner import hydra_runner
import common.no_warning # noqa pylint: disable=W0611
from common.utils import encode_etlt

from cv.efficientdet.config.default_config import ExperimentConfig
from cv.efficientdet.exporter.onnx_exporter import EfficientDetGraphSurgeon
from cv.efficientdet.inferencer import inference
from cv.efficientdet.utils import helper, hparams_config
from cv.efficientdet.utils.config_utils import generate_params_from_cfg

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def setup_env():
    """Setup export env."""
    tf.autograph.set_verbosity(0)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


@monitor_status(name='efficientdet', mode='export')
def run_export(cfg):
    """Launch EfficientDet export."""
    # Parse and update hparams
    MODE = 'export'
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode=MODE))
    params = config.as_dict()

    assert str(cfg.export.output_path).endswith('.etlt'), "Exported file must end with .etlt"
    output_dir = tempfile.mkdtemp()  # os.path.dirname(cfg.export.output_path)
    tf.keras.backend.set_learning_phase(0)

    # Load model from graph json
    model = helper.load_model(cfg.export.model_path, cfg, MODE, is_qat=cfg.train.qat)
    model.summary()
    # Get input shape from model
    input_shape = list(model.layers[0].input_shape[0])
    max_batch_size = 1 if cfg.export.dynamic_batch_size else cfg.export.max_batch_size
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
    onnx_file = effdet_gs.save()
    encode_etlt(onnx_file, cfg.export.output_path, "", cfg.key)
    # print(f"The exported model is saved at: {onnx_file}")
    os.remove(onnx_file)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="export", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """Wrapper function for EfficientDet exporter."""
    setup_env()
    run_export(cfg=cfg)


if __name__ == '__main__':
    main()

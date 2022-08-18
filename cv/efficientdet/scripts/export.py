# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Export EfficientDet model to etlt and TRT engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.util import deprecation

from common.hydra.hydra_runner import hydra_runner

from cv.efficientdet.config.default_config import ExperimentConfig
from cv.efficientdet.exporter.onnx_exporter import EfficientDetGraphSurgeon
from cv.efficientdet.exporter.trt_builder import EngineBuilder
from cv.efficientdet.inferencer import inference
from cv.efficientdet.utils import helper, hparams_config
from cv.efficientdet.utils.config_utils import generate_params_from_cfg

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def run_export(cfg):
    """Launch EfficientDet export."""
    # disable_eager_execution()
    tf.autograph.set_verbosity(0)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    # Parse and update hparams
    MODE = 'export'
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode='export'))
    params = config.as_dict()

    # TODO(@yuw): change to EFF
    assert str(cfg.export.output_path).endswith('.onnx'), "ONNX!!!"
    output_dir = os.path.dirname(cfg.export.output_path)
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

    print("Generating ONNX model...")
    # convert to onnx
    effdet_gs = EfficientDetGraphSurgeon(
        output_dir,
        legacy_plugins=False,
        dynamic_batch=cfg.export.dynamic_batch_size,
        is_qat=cfg.train.qat)
    effdet_gs.update_preprocessor('NHWC', input_shape, preprocessor="imagenet")
    effdet_gs.update_shapes()
    effdet_gs.update_nms(threshold=cfg.export.min_score_thresh)
    # TODO(@yuw): convert onnx to eff
    onnx_file = effdet_gs.save(cfg.export.output_path)
    print("Generating TensorRT engine...")
    # convert to engine
    if cfg.export.engine_file is not None or cfg.export.data_type == 'int8':

        output_engine_path = cfg.export.engine_file
        builder = EngineBuilder(cfg.verbose, workspace=cfg.export.max_workspace_size)
        builder.create_network(onnx_file, batch_size=max_batch_size)
        builder.create_engine(
            output_engine_path,
            cfg.export.data_type,
            cfg.export.cal_image_dir,
            cfg.export.cal_cache_file,
            cfg.export.cal_batch_size * cfg.export.cal_batches,
            cfg.export.cal_batch_size)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="export", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """Wrapper function for EfficientDet exporter."""
    run_export(cfg=cfg)


if __name__ == '__main__':
    main()

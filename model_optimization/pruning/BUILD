# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

load("//moduluspy/bazel:python_helpers.bzl", "maglev_library")
load("//ci/runtime_resources:extension.bzl", "py_import_test", "py_static_analysis", "pytest_test")
load("@pip_deps//:requirements.bzl", "requirement")

maglev_library(
    name = "pruning",
    srcs = glob(
        ["*.py"],
        exclude = ["test_*.py"],
    ),
    deps = [
        "//moduluspy/modulus/decorators",
        "//moduluspy/modulus/models/templates",
        "//moduluspy/modulus/models:import_keras",
        "//iva/retinanet/initializers:prior_prob",
        "//iva/retinanet/layers:anchor_box_layer",
        "//iva/ssd/layers:anchor_box_layer",
        "//iva/templates:utils",
        "//iva/yolo_v3/layers:yolo_anchor_box_layer",
        "//iva/yolo_v4/layers:bbox_postprocessing_layer",
        "//iva/yolo_v4/layers:split",
        "//iva/faster_rcnn/layers:custom_layers",
        "//iva/mask_rcnn/layers:anchor_layer",
        "//iva/mask_rcnn/layers:reshape_layer",
        "//iva/mask_rcnn/layers:box_target_encoder",
        "//iva/mask_rcnn/layers:foreground_selector_for_mask",
        "//iva/mask_rcnn/layers:gpu_detection_layer",
        "//iva/mask_rcnn/layers:mask_targets_layer",
        "//iva/mask_rcnn/layers:mask_postprocess_layer",
        "//iva/mask_rcnn/layers:multilevel_crop_resize_layer",
        "//iva/mask_rcnn/layers:multilevel_proposal_layer",
        "//iva/mask_rcnn/layers:proposal_assignment_layer",
        "//iva/mask_rcnn/layers:image_input_layer",
        "//iva/mask_rcnn/layers:info_input_layer",
        "//iva/mask_rcnn/layers:class_input_layer",
        "//iva/mask_rcnn/layers:box_input_layer",
        "//iva/mask_rcnn/layers:mask_input_layer",
        "//iva/efficientdet/layers:image_resize_layer",
        "//iva/efficientdet/layers:weighted_fusion_layer",
        "//iva/efficientdet/utils:utils",
        "//iva/templates:utils_tf",
        requirement("keras"),
        requirement("h5py"),
        requirement("tensorflow"),
    ],
)

py_import_test(
    name = "pruning",
    dotted_name = "modulus.pruning",
)

# test_pruning.py will be run in both keras and tf.keras.
test_pruning_kwargs = {
    "filename": "test_pruning.py",
    "tags": ["needs_gpu"],
    "deps": [
        ":pruning",
        "//moduluspy:modulus-prune",
        "//moduluspy/modulus/models:import_keras",
        "//moduluspy/modulus/models/templates",
        requirement("pytest-console-scripts"),
    ],
    "shard_count": 5,
    "timeout": "long",
    "config": "//moduluspy:test_config",
}

pytest_test(
    name = "test_pruning",
    timeout = test_pruning_kwargs["timeout"],
    config = test_pruning_kwargs["config"],
    filename = test_pruning_kwargs["filename"],
    shard_count = test_pruning_kwargs["shard_count"],
    tags = test_pruning_kwargs["tags"],
    deps = test_pruning_kwargs["deps"],
)

pytest_test(
    name = "test_pruning_tf_keras",
    timeout = test_pruning_kwargs["timeout"],
    config = test_pruning_kwargs["config"],
    env_vars = {"TF_KERAS": "1"},
    filename = test_pruning_kwargs["filename"],
    shard_count = test_pruning_kwargs["shard_count"],
    tags = test_pruning_kwargs["tags"],
    deps = test_pruning_kwargs["deps"],
)

exports_files(
    ["app.py"],
    visibility = ["//visibility:public"],
)

py_static_analysis(
    name = "static_test",
)

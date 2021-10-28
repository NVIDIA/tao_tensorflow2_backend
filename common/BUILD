# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis", "pytest_test")
load("@pip_deps//:requirements.bzl", "requirement")
load("//maglev_sdk/docker_container:rules.bzl", "container_image_py")
load("//maglev_sdk/launcher:launcher_rules.bzl", "launcher_run")

package(default_visibility = ["//visibility:public"])

container_image_py(
    name = "image",
    srcs = [
        "//iva/common:__init__.py",
    ],
    deps = [
        "//iva/detectnet_v2/docker:detectnet_v2",
        "//iva/faster_rcnn/docker:faster_rcnn",
        "//iva/makenet/docker:makenet",
        "//iva/retinanet/docker:retinanet",
        "//iva/ssd/docker:ssd",
        "//iva/yolo_v3/docker:yolo_v3",
    ],
    add_to_path = [
        "iva/internal/detectnet_v2/scripts",
        "iva/internal/detectnet_v2/docker",
        "iva/detectnet_v2/docker",
        "iva/detectnet_v2/scripts",
        "iva/faster_rcnn/docker",
        "iva/faster_rcnn/scripts",
        "iva/makenet/docker",
        "iva/makenet/scripts",
        "iva/ssd/docker",
        "iva/ssd/scripts",
        "iva/retinanet/docker",
        "iva/retinanet/scripts",
        "iva/yolo_v3/docker",
        "iva/yolo_v3/scripts",
        # Support invocation syntax: "train -h"
    ],
    env = {
        "DATA_BASE_PATH": "/datasets",
    },
    main = "__init__.py",
)

launcher_run(
    name = "run",
    image = ":image",
)

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":magnet_evaluate",
        ":magnet_infer",
        ":magnet_prune",
        ":magnet_train",
        ":magnet_tensorfile",
        ":utils",
        ":spec_validator",
        "no_warning",
        "//iva/common/export:all",
    ]
)

py_binary(
    name = "magnet_infer",
    srcs = ["magnet_infer.py"],
    deps = [
        "//iva/detectnet_v2/scripts:inference",
        "//iva/makenet/scripts:inference",
        "//iva/faster_rcnn/scripts:inference",
        "//iva/ssd/scripts:inference",
        "//iva/yolo_v3/scripts:inference",
        "//iva/retinanet/scripts:inference",
        "//iva/mask_rcnn/scripts:inference",
        "//iva/mask_rcnn/scripts:inference_trt",
        ":no_warning",
        ":utils"
    ]
)

py_binary(
    name = "magnet_train",
    srcs = ["magnet_train.py"],
    deps = [
        "//iva/detectnet_v2/scripts:train",
        "//iva/makenet/scripts:train",
        "//iva/faster_rcnn/scripts:train",
        "//iva/ssd/scripts:train",
        "//iva/yolo_v3/scripts:train",
        "//iva/retinanet/scripts:train",
        "//iva/mask_rcnn/scripts:train",
        ":no_warning",
        ":utils"
    ]
)

py_binary(
    name = "magnet_tensorfile",
    srcs = ["magnet_tensorfile.py"],
    deps = [
        "//iva/detectnet_v2/scripts:calibration_tensorfile",
        "//iva/makenet/scripts:calibration_tensorfile",
        ":no_warning",
        ":utils"
    ]
)

py_binary(
    name = "magnet_evaluate",
    srcs = ["magnet_evaluate.py"],
    deps = [
        "//iva/detectnet_v2/scripts:evaluate",
        "//iva/makenet/scripts:evaluate",
        "//iva/faster_rcnn/scripts:evaluate",
        "//iva/ssd/scripts:evaluate",
        "//iva/yolo_v3/scripts:evaluate",
        "//iva/retinanet/scripts:evaluate",
        "//iva/mask_rcnn/scripts:evaluate",
        ":no_warning",
        ":utils"
    ]
)

py_binary(
    name = "magnet_prune",
    srcs = ["magnet_prune.py"],
    deps = [
        "//iva/common:utils",
        "//moduluspy/modulus",
        ":no_warning",
        "//iva/yolo_v4/layers:split",
    ]
)

py_binary(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//moduluspy/modulus",
        "//magnet/encoding:encoding",
        "//iva/faster_rcnn/layers:custom_layers",
        "//iva/ssd/layers:anchor_box_layer",
        "//iva/retinanet/initializers:prior_prob",
        "//iva/retinanet/layers:anchor_box_layer",
        "//iva/yolo_v3/layers:yolo_anchor_box_layer",
        "//iva/yolo_v4/layers:bbox_postprocessing_layer",
        "//iva/yolo_v4/layers:split",
        "//iva/templates",
        requirement("keras"),
    ],
)

py_library(
    name = "spec_validator",
    srcs = ["spec_validator.py"],
    deps = [
        requirement("protobuf"),
        requirement("six"),
    ],
)

py_library(
    name = "no_warning",
    srcs = ["no_warning.py"],
    deps = [
        requirement("tensorflow"),
        requirement("keras")
    ],
)

pytest_test("test_utils.py",
    deps = [":utils",
            "//moduluspy/modulus",
            requirement("keras"),
            requirement("horovod"),
            "//iva/templates"])


py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

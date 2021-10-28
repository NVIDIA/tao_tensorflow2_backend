# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis", "pytest_test")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "augmentation_lib",
    srcs = ["augmentation_lib.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python"),
    ]
)

py_library(
    name = "base_data_sequence",
    srcs = ["base_data_sequence.py"],
    deps = [
        requirement("numpy"),
        requirement("Pillow"),
        requirement("keras"),
        requirement("tensorflow")
    ]
)

py_library(
    name = "detection_data_sequence",
    srcs = ["detection_data_sequence.py"],
    deps = [
        ":base_data_sequence",
        requirement("opencv-python")
    ]
)

py_binary(
    name = "coco_data_sequence",
    srcs = ["coco_data_sequence.py"],
    deps = [
        ":base_data_sequence",
        requirement("opencv-python"),
        requirement("pycocotools-fix")
    ]
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

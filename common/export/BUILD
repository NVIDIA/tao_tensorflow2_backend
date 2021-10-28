# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")
package(default_visibility = ["//visibility:public"])

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":app",
        ":base_exporter",
        ":base_calibrator",
        ":keras_exporter",
        ":tensorfile",
        ":tensorfile_calibrator",
        ":trt_utils",
    ],
)

py_library(
    name = "app",
    srcs = ["app.py"],
    deps=[
        "//third_party/keras",
        "//iva/common/logging:logging",
        requirement("tensorflow"),
        requirement("keras"),
    ],
)

py_library(
    name = "keras_exporter",
    srcs = ["keras_exporter.py"],
    deps=[
        ":base_exporter",
        ":tensorfile_calibrator",
        "//iva/common:utils",
        "//iva/common/types:base_ds_config",
        "//iva/faster_rcnn/layers:custom_layers",
        requirement("tqdm"),
        requirement("pillow")
    ],
)

py_library(
    name = "base_exporter",
    srcs = ["base_exporter.py"],
    deps=[
        ":tensorfile",
        ":tensorfile_calibrator",
        ":trt_utils",
        "//iva/common/logging:logging",
        requirement("six"),
        requirement("tqdm"),
        requirement("pillow")
    ],
)

py_library(
    name = "base_calibrator",
    srcs = ["base_calibrator.py"],
    deps = []
)

py_library(
    name = "tensorfile_calibrator",
    srcs = ["tensorfile_calibrator.py"],
    deps = [
        ":base_calibrator",
        requirement("pycuda"),
        requirement("numpy"),
        ":tensorfile"
    ]
)

py_library(
    name = "tensorfile",
    srcs = ["tensorfile.py"],
    deps = [
        requirement("h5py"),
        requirement("numpy"),
    ]
)

py_library(
    name = "trt_utils",
    srcs = ["trt_utils.py"],
    deps = [
        requirement("tensorrt"),
    ]
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "engine",
    srcs = ["engine.py"],
    deps = [
        requirement("numpy"),
        requirement("pycuda"),
    ],
)

py_library(
    name = "trt_inferencer",
    srcs = ["trt_inferencer.py"],
    deps = [
        ":engine",
        requirement("numpy"),
        requirement("tensorrt")
    ],
)

py_library(
    name = "inferencer",
    srcs = ["inferencer.py"],
    deps = [
        ":trt_inferencer",
        "//iva/makenet/utils:preprocess_input",
        requirement('Pillow'),
        requirement('tqdm'),
        requirement('keras'),
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

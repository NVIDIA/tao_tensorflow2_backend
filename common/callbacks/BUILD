# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_metric_callback",
    srcs = ["base_metric_callback.py"],
    deps = [
        requirement("keras")
    ],
)

py_library(
    name = "detection_metric_callback",
    srcs = ["detection_metric_callback.py"],
    deps = [
        ":base_metric_callback",
        requirement("keras"),
        requirement("tqdm"),
        requirement("numpy")
    ],
)

py_library(
    name = "enc_model_saver_callback",
    srcs = ["enc_model_saver_callback.py"],
    deps = [
        "//magnet/encoding:encoding",
        requirement("keras")
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

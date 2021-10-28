# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_loss",
    srcs = ["base_loss.py"],
    deps = [
        requirement('tensorflow')
    ],
)
py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

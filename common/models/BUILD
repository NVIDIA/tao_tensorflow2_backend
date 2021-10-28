# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "backbones",
    srcs = ["backbones.py"],
    deps = [
        "//iva/templates",
    ]
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    deps = [
        "//iva/templates",
        requirement('keras'),
    ]
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

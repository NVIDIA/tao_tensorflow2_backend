# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")
package(default_visibility = ["//visibility:public"])

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "logging",
    srcs = ["logging.py"],
    deps=[],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

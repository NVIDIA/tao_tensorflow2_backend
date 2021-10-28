# Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "parallelize_model",
    srcs = ["parallelize_model.py"],
    deps = [
        requirement("tensorflow"),
        requirement("keras"),
        requirement("numpy"),
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "makenet",
    srcs = ["__init__.py"],
    deps = [
        "//iva/makenet/model:model_builder",
        "//iva/makenet/scripts:all",
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

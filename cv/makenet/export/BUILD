# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":classification_exporter",
    ],
)

py_binary(
    name = "classification_exporter",
    srcs = ["classification_exporter.py"],
    deps=[
        "//iva/common/export:keras_exporter",
        "//iva/common/types:base_ds_config",
        "//moduluspy/modulus/export",
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

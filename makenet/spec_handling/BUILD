# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":constants",
        ":spec_loader"
    ]
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [
        "//iva/common:spec_validator",
    ],
)

py_library(
    name = "spec_loader",
    srcs = ["spec_loader.py"],
    deps = [
        "//iva/makenet/proto:experiment_pb2",
        "//iva/common:spec_validator",
        ":constants",
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

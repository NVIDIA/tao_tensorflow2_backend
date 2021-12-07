# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":helper",
        ":preprocess_crop",
        ":preprocess_input",
        ":callbacks",
    ]
)

py_library(
    name = "helper",
    srcs = ["helper.py"],
    deps = [
        "//moduluspy/modulus",
        "//iva/common:utils",
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("numba"),
        requirement("Pillow"),
    ]
)

py_library(
    name = "preprocess_crop",
    srcs = ["preprocess_crop.py"],
    deps = [
        requirement("keras"),
    ]
)

py_library(
    name = "callbacks",
    srcs = ["callbacks.py"],
    deps = [
        "//iva/common:utils",
        "//moduluspy/modulus",
    ],
)

py_library(
    name = "preprocess_input",
    srcs = ["preprocess_input.py"],
    deps = [
        requirement("numpy"),
        requirement("keras"),
    ]
)

py_library(
    name = "mixup_generator",
    srcs = ["mixup_generator.py"],
    deps = [
        requirement("numpy"),
    ]
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis", "pytest_test")
load("@pip_deps//:requirements.bzl", "requirement")
package(default_visibility = ["//visibility:public"])


pytest_test("test_export_tools.py",
    deps = [
        requirement("tensorflow"),
        requirement("tensorrt"),
        requirement("uff"),
        requirement("graphsurgeon"),
        requirement("pytest"),
        ])


py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

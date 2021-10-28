# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
load("@pip_deps//:requirements.bzl", "requirement")
load("//ci/runtime_resources:extension.bzl", "py_static_analysis", "pytest_test")
package(default_visibility = ["//visibility:public"])

pytest_test("test_freeze_bn.py",
    deps = [
        requirement("keras"),
        requirement("pytest"),
        "//iva/makenet/model:model_builder",
        "//iva/makenet/scripts:train",
    ],
)


py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

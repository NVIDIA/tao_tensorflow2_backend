# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

load("//ci/runtime_resources:extension.bzl", "py_static_analysis")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "all",
    srcs = ["__init__.py"],
    deps = [
        ":calibration_tensorfile",
        ":evaluate",
        ":inference",
        ":prune",
        ":export",
        ":train",
    ],
)

py_binary(
    name = "calibration_tensorfile",
    srcs = ["calibration_tensorfile.py"],
    deps = [
         "//iva/makenet/spec_handling:spec_loader",
         "//iva/makenet/utils:preprocess_crop",
         "//iva/makenet/utils:preprocess_input",
         "//moduluspy/modulus",
         requirement("numpy"),
         requirement('pillow'),
         requirement('tqdm'),
    ]

)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
         "//iva/makenet/spec_handling:spec_loader",
         "//iva/makenet/utils:helper",
         "//iva/common:utils",
         "//iva/makenet/utils:preprocess_input",
         "//iva/makenet/utils:preprocess_crop",
         requirement('pillow'),
         requirement('scikit-learn')
    ]
)

py_binary(
    name = "inference",
    srcs = ["inference.py"],
    deps = [
        "//moduluspy/modulus",
        "//iva/common:utils",
        "//iva/makenet/utils:helper",
        "//iva/makenet/spec_handling:spec_loader",
        "//iva/makenet/utils:preprocess_input",
        "//iva/makenet/utils:preprocess_crop",
        requirement("tqdm"),
    ],
)

py_binary(
    name = "prune",
    srcs = ["prune.py"],
    deps = [
            "//iva/common:magnet_prune",
           ],
)

py_binary(
    name = "export",
    srcs = ["export.py"],
    deps = [
            "//iva/makenet/export:classification_exporter",
            "//iva/common/export:app",
           ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "//iva/makenet/model:model_builder",
        "//iva/common:utils",
        "//iva/common/model_parallelism:parallelize_model",
        "//iva/makenet/spec_handling:spec_loader",
        "//iva/makenet/utils:callbacks",
        "//iva/makenet/utils:helper",
        "//iva/makenet/utils:preprocess_crop",
        "//iva/makenet/utils:preprocess_input",
        "//iva/makenet/utils:mixup_generator",
        requirement('pillow'),
    ]
)

sh_binary(
    name = "train_multigpu",
    srcs = ["train_multigpu.sh"],
    data = [
        "//iva/makenet/scripts:train",
    ],
)

py_static_analysis(
    name = "static_test",
    python_module = "iva",
)

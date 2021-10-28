# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
# Build rules for compiling DriveNet experiment protos.

load("//tools/proto:rules.bzl", "py_proto_library")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "detection_sequence_dataset_config_pb2",
    protos = ["detection_sequence_dataset_config.proto"],
)

py_proto_library(
    name = "adam_optimizer_config_pb2",
    protos = ["adam_optimizer_config.proto"],
)

py_proto_library(
    name = "sgd_optimizer_config_pb2",
    protos = ["sgd_optimizer_config.proto"],
)

py_proto_library(
    name = "rmsprop_optimizer_config_pb2",
    protos = ["rmsprop_optimizer_config.proto"],
)

py_proto_library(
    name = "optimizer_config_pb2",
    protos = ["optimizer_config.proto"],
    proto_deps = [
        ":rmsprop_optimizer_config_pb2",
        ":sgd_optimizer_config_pb2",
        ":adam_optimizer_config_pb2"
    ],
)

py_proto_library(
    name = "cost_scaling_config_pb2",
    protos = ["cost_scaling_config.proto"],
)

py_proto_library(
    name = "soft_start_annealing_schedule_config_pb2",
    protos = ["soft_start_annealing_schedule_config.proto"],
)

py_proto_library(
    name = "soft_start_cosine_annealing_schedule_config_pb2",
    protos = ["soft_start_cosine_annealing_schedule_config.proto"],
)

py_proto_library(
    name = "regularizer_config_pb2",
    protos = ["regularizer_config.proto"],
)

py_proto_library(
    name = "learning_rate_config_pb2",
    protos = ["learning_rate_config.proto"],
    proto_deps = [
        ":soft_start_annealing_schedule_config_pb2",
        ":soft_start_cosine_annealing_schedule_config_pb2"
    ],
)

py_proto_library(
    name = "training_config_pb2",
    protos = ["training_config.proto"],
    proto_deps = [
        ":learning_rate_config_pb2",
        ":cost_scaling_config_pb2",
        ":optimizer_config_pb2",
        ":regularizer_config_pb2"
    ],
)

py_proto_library(
    name = "eval_config_pb2",
    protos = ["eval_config.proto"],
)

py_proto_library(
    name = "nms_config_pb2",
    protos = ["nms_config.proto"],
)

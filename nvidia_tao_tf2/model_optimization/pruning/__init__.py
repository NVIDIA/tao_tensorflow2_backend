# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Modulus pruning APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf2.model_optimization.pruning import pruning
from nvidia_tao_tf2.model_optimization.pruning.pruning import prune

__all__ = ("prune", "pruning")

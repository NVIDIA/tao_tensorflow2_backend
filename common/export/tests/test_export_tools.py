# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Simple script to test export tools(tensorrt, uff, graphsurgeon)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graphsurgeon as gs
import tensorrt as trt
import uff


# Check gs has the create_plugin_node method
def test_gs_create_plugin_node():
    n = gs.create_plugin_node(name='roi_pooling_conv_1/CropAndResize_new',
                              op="CropAndResize",
                              inputs=['activation_13/Relu', 'proposal'],
                              crop_height=7,
                              crop_width=7)
    assert n


# Check the TRT version
def test_trt_version():
    assert trt.__version__ == '8.0.1.6'


# Check the UFF version
def test_uff_version():
    assert uff.__version__ == '0.6.7'


# Check the gs version
def test_gs_version():
    assert gs.__version__ == '0.4.1'

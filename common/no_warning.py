# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Include this in wrapper to suppress all warnings."""
# Code below to suppress as many warnings as possible
import os
if str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '1':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger('tensorflow').disabled = True
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.FATAL)

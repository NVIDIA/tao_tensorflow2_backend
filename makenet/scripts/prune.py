# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the classification TLT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from iva.common.magnet_prune import (  # noqa pylint: disable=unused-import
    build_command_line_parser,
    main,
)


if __name__ == "__main__":
    main(sys.argv[1:])

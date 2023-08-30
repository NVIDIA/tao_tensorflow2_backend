# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decorator for experimental logging."""

import logging

# Configure the logger.
logging.basicConfig(
    format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
    level="INFO"
)
logger = logging.getLogger(__name__)

def experimental(fn):
    """Simple function to define the keras decorator.

    This decorator clears any previously existing sessions
    and sets up a new session.
    """
    def _fn_wrapper(*args, **kwargs):
        """Clear the keras session."""
        logger.warning("This is an experimental module. Please use this at your own risk.")
        return fn(*args, **kwargs)
    return _fn_wrapper

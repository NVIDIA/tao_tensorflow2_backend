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

"""Setup script to build the TAO Toolkit package."""

import os
import setuptools

from release.python.utils import utils

PACKAGE_LIST = [
    "nvidia_tao_tf2"
]

version_locals = utils.get_version_details()
setuptools_packages = []
for package_name in PACKAGE_LIST:
    setuptools_packages.extend(utils.find_packages(package_name))

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license=version_locals['__license__'],
    keywords=version_locals['__keywords__'],
    packages=setuptools_packages,
    package_data={
        '': ['*.py', "*.pyc", "*.yaml", "*.so", "*.pdf"]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'classification_tf2=nvidia_tao_tf2.cv.classification.entrypoint.classification:main',
            'efficientdet_tf2=nvidia_tao_tf2.cv.efficientdet.entrypoint.efficientdet:main',
        ]
    }
)

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Setup script to build the TAO Toolkit package."""

import os
import setuptools

from release.python.utils import utils


version_locals = utils.get_version_details()
found_packages = setuptools.find_packages(
    include=(
        "backbones", "blocks",
        "common", "cv",
        "model_optimization"
    )
)

setuptools.setup(
    name='nvidia-tao-tf2',
    version=version_locals["__version__"],
    description="NVIDIA's package for DNN implementation on TensorFlow 2.x for use with TAO Toolkit.",
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        # Pick your license as you wish (should match "license" above)
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license="NVIDIA Proprietary Software",
    keywords='tao',
    packages=found_packages,
    package_data={
        '': ['*.py', "*.pyc", "*.yaml", "*.so"]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'classification=cv.classification.entrypoint.classification:main',
            'efficientdet=cv.efficientdet.entrypoint.efficientdet:main',
        ]
    }
)

# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

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

if(os.path.exists("pytransform_vax_001219")):
    pyarmor_packages = ["pytransform_vax_001219"]
    setuptools_packages += pyarmor_packages

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
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

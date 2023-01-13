"""
setup.py

unstructured_inference - Tools to utilize trained models

Copyright 2022 Unstructured Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, find_packages

from unstructured_inference.__version__ import __version__

setup(
    name="unstructured_inference",
    description="A library for performing inference using trained models.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP PDF HTML CV XML parsing preprocessing",
    url="https://github.com/Unstructured-IO/unstructured-inference",
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author="Unstructured Technologies",
    author_email="devops@unstructuredai.io",
    license="Apache-2.0",
    packages=find_packages(),
    version=__version__,
    entry_points={},
    install_requires=[
        "fastapi",
        "layoutparser[layoutmodels,tesseract]",
        "python-multipart",
        "uvicorn",
        "huggingface-hub",
        # NOTE(robinson) - later versions of opencv-python cause installation issues
        # on RHEL7. We can remove this pin once the following issue from 12/2022 is resolved
        # ref: https://github.com/opencv/opencv-python/issues/772
        "opencv-python==4.6.0.66",
    ],
    extras_require={},
)

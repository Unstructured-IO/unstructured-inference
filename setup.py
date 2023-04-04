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


def load_requirements(file_list=None):
    if file_list is None:
        file_list = ["requirements/base.in"]
    if isinstance(file_list, str):
        file_list = [file_list]
    requirements = []
    for file in file_list:
        if not file.startswith("#"):
            with open(file, encoding="utf-8") as f:
                requirements.extend(f.readlines())
    return requirements


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
    install_requires=load_requirements(),
    extras_require={
        "tables": [
            'unstructured.PaddleOCR ; platform_machine=="x86_64"',
            # NOTE(crag): workaround issue for error output below
            # ERROR test_unstructured/partition/test_common.py - TypeError: Descriptors cannot not
            # be created directly.
            # If this call came from a _pb2.py file, your generated code is out of date and must be
            # regenerated with protoc >= 3.19.0.
            # If you cannot immediately regenerate your protos, some other possible workarounds are:
            #  1. Downgrade the protobuf package to 3.20.x or lower.
            #  2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python
            #     parsing and will be much slower).
            'protobuf<3.21 ; platform_machine=="x86_64"',
            # NOTE(alan): Pin to get around error: undefined symbol: _dl_sym, version GLIBC_PRIVATE
            'paddlepaddle>=2.4 ; platform_machine=="x86_64"',
        ]
    },
)

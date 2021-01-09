# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""
Setup script for this Python package.
https://docs.python.org/3/distutils/setupscript.html
"""


import pathlib
from setuptools import setup
from confidential_ml_utils import __version__


HERE = pathlib.Path(__file__).parent

README = (HERE / ".." / "README.md").read_text()

setup(
    name="confidential-ml-utils",
    version=__version__,
    description="Utilities for confidential machine learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Azure/confidential-ml-utils",
    author="AML Data Science",
    author_email="aml-ds@microsoft.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=["confidential_ml_utils"],
    include_package_data=True,
    install_requires=[],
    # https://stackoverflow.com/a/48777286
    python_requires="~=3.6",
)

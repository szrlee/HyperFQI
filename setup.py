#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="hyperfqi",
    version="0.0.1",
    python_requires=">=3.6",
    install_requires=[
        "gym",
        "tqdm",
        "numpy>1.16.0",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard>=2.5.0",
        "torch==1.8.1",
        "numba>=0.51.0",
        "h5py>=2.10.0",  # to match tensorflow's minimal requirements
    ],
    packages=find_packages(
        exclude=["experiments", "atari_results"]
    ),
    extras_require={"atari": ["atari_py", "opencv-python"]
    },
)

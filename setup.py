#!/usr/bin/env python
from distutils.core import setup

setup(
    name='Streamer',
    version='1.0',
    description='Using pulsar and tensorflow as the backend of a twitter sentiment app',
    install_requires=[
        'tensorflow',
        'keras',
        'keras_preprocessing',
        'pandas',
        'tqdm',
        'sklearn',
    ],
)

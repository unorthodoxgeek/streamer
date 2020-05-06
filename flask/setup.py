#!/usr/bin/env python
from distutils.core import setup

setup(
    name='Flask Modeler',
    version='1.0',
    description='Serving a ML model via a simple API',
    install_requires=[
        'keras',
        'keras_preprocessing',
        'flask',
        'requests',
        'tensorflow',
    ],
)

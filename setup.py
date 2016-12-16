#!/usr/bin/env python2

from setuptools import setup, find_packages
from os.path import join, dirname

setup(
	name='3d_segment_nuclei',
	version='0.1',
	packages=find_packages(),
	long_description=open( join(dirname(__file__), 'README.txt')).read(),
)

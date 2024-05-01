#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='datageneration',
      version='0.1',
      long_description_content_type='text/markdown',
      author='Asad Hussain, Sharan Banagiri, Jacob Golomb, Alexander Criswell',
      author_email='asadh@utexas.edu',
      url='https://github.com/Potatoasad/gb-population-inference',
      license='MIT',
      packages=find_packages(exclude=('tests', 'docs', 'dev')),
      package_data={'datageneration': ['data/*']},
      install_requires=[
            'h5py',
            'matplotlib',
            'numpy',
            'pandas',
            'tqdm',
            'scipy',
            'pandas'
            ]
     )
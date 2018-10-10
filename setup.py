import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

scripts = [
    'scripts/image-fft',
    'scripts/image-fft-tk',
]
setup(
    name='image-fft',
    version='0.1',
    description='Image FFT',
    author='Michael Lamparski',
    author_email='diagonaldevice@gmail.com',
    packages=['image_fft'],
    install_requires=['numpy', 'Pillow', 'networkx'], # 'pyfftw'
    provides=['image_fft'],
    scripts=scripts,
)

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def open_file(fname):
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(
  name = 'ader',
  packages = ['ader', 'ader.etc', 'ader.dg', 'ader.fv', 'ader.weno'],
  version = '1.1.1',
  description = 'The ADER method for solving any (potentially very stiff) hyperbolic system of PDEs',
  long_description=open_file('README.rst').read(),
  author = 'Haran Jackson',
  author_email = 'jackson.haran@gmail.com',
  license="MIT",
  url = 'https://github.com/haranjackson/ADER',
  download_url = 'https://github.com/haranjackson/ADER/archive/1.1.1.tar.gz',
  keywords = ['ADER', 'WENO', 'Discontinuous Galerkin', 'Finite Volume', 'PDEs', 'Partial Differential Equations'],
  classifiers=[
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Atmospheric Science',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics'
  ],
  install_requires=['numpy>=1.13', 'scipy>=0.19', 'tangent>=0.1.9']
)

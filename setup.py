#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("find_grads", ["bug_model/find_grads.pyx"],
                                  include_dirs=[np.get_include()])]   # <---- New argument

setup(
  name = 'bug_model',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  packages = ['bug_model', 'bug_model.header']
)

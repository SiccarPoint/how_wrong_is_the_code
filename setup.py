from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("find_grads", ["find_grads.pyx"],
                                  include_dirs=[np.get_include()])]   # <---- New argument

setup(
  name = 'how_wrong_is_the_code',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

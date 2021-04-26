from distutils.core import setup
from distutils.extension import Extension

import Cython.Build
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension('pair_potential.lj_potential_c',
              sources=['./pair_potential/lj_potential_c.pyx'],
              libraries=['m'],
              )
]

setup(
    name='computational physics',
    packages=['pair_potential'],
    ext_modules=Cython.Build.cythonize(ext_modules, language_level=3)
)

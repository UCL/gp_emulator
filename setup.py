#!/usr/bin/env python
from distutils.core import setup

from setuptools import setup
import distutils.command.build as _build
import setuptools.command.install as _install

import sys
import os
import os.path as op
import distutils.spawn as ds
import distutils.dir_util as dd



def run_cmake():
    if ds.find_executable('cmake') is None:
        print "CMake  is required"
        print "Please install cmake version >= 2.6 and re-run setup"
        sys.exit(-1)

    build_dir = op.join(op.split(__file__)[0], 'build')
    dd.mkpath(build_dir)
    os.chdir(build_dir)

    try:
        ds.spawn(['cmake','../'])
    except ds.DistutilsExecError:
        print "Error while running cmake"
        sys.exit(-1)
    try:
        ds.spawn(['make','-j'])
    except ds.DistutilsExecError:
        print "Error while compiling"
        sys.exit(-1)
 

class install(_install.install):
    def run(self):
        cwd = os.getcwd()
        run_cmake()
        os.chdir(cwd)
        _install.install.run(self)

class build(_build.build):
    def run(self):
        cwd = os.getcwd()
        run_cmake()
        os.chdir(cwd)
        _build.build.run(self)
class test(_install.install):
    def run(self):
        os.system("python tests/benchmark.py")



setup(name='gp_emulator',
      version='1.4.3',
      description='A Python GaussianProcess emulator software package',
      author='J Gomez-Dans',
      author_email='j.gomez-dans@ucl.ac.uk',
      url='http://bitbucket.org/gomezdansj/gp_emulator',
      packages=['gp_emulator'],
      cmdclass={'build':build, 'install':install, 'test':test},
     )

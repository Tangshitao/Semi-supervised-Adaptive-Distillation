# Copyright 2014 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from subprocess import Popen


this_dir = os.path.dirname(os.path.realpath(__file__))

package_data = {'nervanagpu': ['kernels/sass/*.sass',
                               'kernels/cubin/*.cubin']}

class build_py(_build_py):
    """
    Specialized source builder required to compile and update CUDA kernels.
    """
    def run(self):
        # compile and patch kernels prior to installation
        sts = Popen("make kernels", shell=True, cwd=this_dir).wait()
        if sts != 0:
            raise OSError(sts, 'Problems compiling kernels')
        _build_py.run(self)

     
setup(
      name='nervanagpu',
      version='0.3.3',
      description='Python bindings for Nervana GPU kernels',
      url='https://github.com/nervanasys/nervanagpu',
      author='Nervana Systems',
      license='(see LICENSE document)',
      install_requires=['numpy',
                        'pycuda>=2015.1',
                        'pytools'],
      packages = find_packages(),
      package_data=package_data,
      cmdclass={'build_py': build_py},
     )

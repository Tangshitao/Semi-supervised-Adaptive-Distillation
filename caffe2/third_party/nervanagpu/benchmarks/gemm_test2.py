#!/usr/bin/python
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

import numpy as np
from nervanagpu import NervanaGPU
from pycuda.autoinit import context

print(context.get_device().name())

np.set_printoptions(threshold=8193, linewidth=600, formatter={'float':lambda x: "% .0f" % x})

ng = NervanaGPU(stochastic_round=False, bench=True)

dtype  = np.float16
repeat = 1
cpu    = 1  # Set CPU to 1 to check against CPU

for data_type in ("All Ones", "Random Data",): #"All Ones", "Random Data"
    print(data_type)
    for size in ((3072,3072,3072*2),): #(4095,4095,4095) 
        m, n, k = size
        for op in ("tn","nn","nt"): #"tn","nn","nt"

            dimA = (m,k) if op[0] == 'n' else (k,m)
            dimB = (k,n) if op[1] == 'n' else (n,k)
            dimC = (m,n)

            if data_type == "All Ones":
                cpuA = np.ones(dimA, dtype=dtype).astype(np.float32)
                cpuB = np.ones(dimB, dtype=dtype).astype(np.float32)
                #cpuB = np.identity(n, dtype=np.float32)
            else:
                cpuA = np.random.uniform(-1.0, 1.0, dimA).astype(np.float32)
                cpuB = np.random.uniform(-1.0, 1.0, dimB).astype(np.float32)

            devA = ng.array(cpuA, dtype=dtype)
            devB = ng.array(cpuB, dtype=dtype)
            devC = ng.empty(dimC, dtype=dtype)

            if op[0] == 't': cpuA, devA = cpuA.T, devA.T
            if op[1] == 't': cpuB, devB = cpuB.T, devB.T

            ng.dot(devA, devB, devC, repeat=repeat)

            if cpu:

                cpuC = np.dot(cpuA, cpuB)

                cpuD = devC.get()
                diff = np.absolute(cpuC - cpuD)

                print(diff.max())
                print(cpuD[::max(m//4,1),::max(n//4,1)])
                print(cpuC[::max(m//4,1),::max(n//4,1)])
                print(diff[::max(m//4,1),::max(n//4,1)])

                # print(cpuD)
                # exit()

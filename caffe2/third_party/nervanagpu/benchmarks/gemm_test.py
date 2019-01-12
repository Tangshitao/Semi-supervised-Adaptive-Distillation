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
from time import sleep

np.set_printoptions(threshold=8193, linewidth=600, formatter={'float':lambda x: "% .1f" % x})

dtype = np.float32

ng = NervanaGPU(stochastic_round=False, bench=False)

small  = (1,2,3,4,5,6,7,8,9,16,32,64,65,72,120,127,128,192)
medium = (1,64,192,778,785,786,787,794)
big    = (1,64,192,1532,1535,1536,1537,1540)

for size in (small,medium,big): #  small, medium, big
    for m in size:
        for n in (size):
            for op in ("tn","nn","nt"): # "tn","nn","nt",
                for k in size:
                    print("op,M,N,K: ", op, m, n, k)

                    dimA = (m,k) if op[0] == 'n' else (k,m)
                    dimB = (k,n) if op[1] == 'n' else (n,k)
                    dimC = (m,n)

                    cpuA = np.random.uniform(-1.0, 1.0, dimA).astype(np.float32)
                    cpuB = np.random.uniform(-1.0, 1.0, dimB).astype(np.float32)
                    #cpuB = np.identity(n, dtype=dtype)

                    devA = ng.array(cpuA, dtype=dtype)
                    devB = ng.array(cpuB, dtype=dtype)
                    devC = ng.empty(dimC, dtype=dtype)

                    #repeat = min(int(50.0 * 4096**3 / (m * n * k)), 1000)

                    if op[0] == 't': cpuA, devA = cpuA.T, devA.T
                    if op[1] == 't': cpuB, devB = cpuB.T, devB.T

                    ng.dot(devA, devB, devC, repeat=1)

                    #context.synchronize()

                    cpuC = np.dot(cpuA, cpuB)

                    cpuD = devC.get()
                    diff = np.absolute(cpuC - cpuD)
                    max_diff = diff.max()
                    print(max_diff, cpuD.max())
                    if max_diff > 0.1 or max_diff != max_diff:
                        #print(m, n, k, max_diff)
                        print(cpuD[::max(m//16,1),::max(n//16,1)])
                        print(cpuC[::max(m//16,1),::max(n//16,1)])
                        print(diff[::max(m//16,1),::max(n//16,1)])
                        exit()

                    # print(max_diff, diff.min(), np.sum(cpuC) - np.sum(cpuD))






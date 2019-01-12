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
import pycuda.driver as drv
from nervanagpu import NervanaGPU
from pycuda.autoinit import context
from scikits.cuda import cublas

print(context.get_device().name())

handle = cublas.cublasCreate()

start, end = (drv.Event(), drv.Event())

def cublas_dot(A, B, C, alpha=1.0, beta=0.0, repeat=1):

    lda = max(A.strides)
    ldb = max(B.strides)
    ldc = max(C.strides)

    opA = 't' if A.is_trans else 'n'
    opB = 't' if B.is_trans else 'n'
    op  = opB + opA

    m = A.shape[0]
    n = B.shape[1]
    k = A.shape[1]

    start.record()
    
    # Swap A and B to map from C order to Fortran 
    for r in range(repeat):
        cublas.cublasSgemm(handle, opB, opA, n, m, k, alpha, B.gpudata, ldb, A.gpudata, lda, beta, C.gpudata, ldc)

    end.record()
    end.synchronize()
    msecs = end.time_since(start) / repeat
    gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
    print("%7.3f msecs %4.0f gflops (%s_%s   : %d,%d,%d)" %
          (msecs,gflops,"cublas",op,m,n,k))


np.set_printoptions(threshold=8193, linewidth=600, formatter={'float':lambda x: "% .0f" % x})

ng = NervanaGPU(stochastic_round=False, bench=True)

repeat = 1

for dtype in (np.float16, np.float32,):
    
    for K, C, N in ((32,4096,1512),):

        for alpha, beta in ((1.0,0.0), (0.5,0.5)):

            for op,  dimA,  dimB,  dimC in (
              ("nn", (K,C), (C,N), (K,N) ),  # fprop
              ("tn", (K,C), (K,N), (C,N) ),  # bprop
              ("nt", (K,N), (C,N), (K,C) ),): # update

                devA1 = ng.empty(dimA, dtype=dtype)
                devB1 = ng.empty(dimB, dtype=dtype)
                devC1 = ng.empty(dimC, dtype=dtype)

                # fill with uniform randoms from -1 to 1
                devA1[:] = 2 * (.5 - ng.rand())
                devB1[:] = 2 * (.5 - ng.rand())
                devC1[:] = 2 * (.5 - ng.rand())

                # just alias if same dtype
                if dtype is np.float32:
                    devA2 = devA1
                    devB2 = devB1
                # otherwise copy
                else:
                    devA2 = ng.empty(dimA, dtype=np.float32)
                    devB2 = ng.empty(dimB, dtype=np.float32)
                    devA2[:] = devA1
                    devB2[:] = devB1

                devC2    = ng.empty(dimC, dtype=np.float32)
                devC2[:] = devC1

                if op[0] == 't': devA1, devA2 = devA1.T, devA2.T
                if op[1] == 't': devB1, devB2 = devB1.T, devB2.T

                ng.dot(devA1, devB1, devC1, alpha=alpha, beta=beta, repeat=repeat)

                cublas_dot(devA2, devB2, devC2, alpha=alpha, beta=beta, repeat=repeat)

                partial1 = ng.empty((devC1.shape[0],1), dtype=np.float32)
                partial2 = partial1[0:1,0:1]

                diff = ng.max(abs(devC2 - devC1), partial=partial1, out=partial2).get()[0,0]
                mean = ng.mean(abs(devC2), partial=partial1, out=partial2).get()[0,0]

                #if diff > .1:
                print("Error: %.3f%%" % (100 * diff / mean))

                print("--------------------------------------------------------------------------------")

cublas.cublasDestroy(handle)

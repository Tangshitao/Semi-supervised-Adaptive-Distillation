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
from ipdb import set_trace

print(context.get_device().name())

handle = cublas.cublasCreate()

start, end = (drv.Event(), drv.Event())

def cublas_dot(A, B, C, repeat=1):

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
        cublas.cublasSgemm(handle, opB, opA, n, m, k, 1.0, B.gpudata, ldb, A.gpudata, lda, 0.0, C.gpudata, ldc)

    end.record()
    end.synchronize()
    msecs = end.time_since(start) / repeat
    gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
    print("%7.3f msecs %4.0f gflops (%s_%s   : %d,%d,%d)" %
          (msecs,gflops,"cublas",op,m,n,k))

    return gflops

np.set_printoptions(threshold=8193, linewidth=600, formatter={'float':lambda x: "% .0f" % x})

ng = NervanaGPU(stochastic_round=False, bench=True)

for dtype in (np.float32, ): #np.float16,
    
    for K, C, N in ((3072,3072*1,64),): 
                    #(3072,3072*1,32),(3072,3072*1,64),(3072,3072*1,96),(3072,3072*1,128),): 
                    #(3072,3072*2,32),(3072,3072*2,64),(3072,3072*2,96),(3072,3072*2,128),
                    #(3072,3072*3,32),(3072,3072*3,64),(3072,3072*3,96),(3072,3072*3,128),
                    #(3072,3072*4,32),(3072,3072*4,64),(3072,3072*4,96),(3072,3072*4,128),
                    #(3072*2,3072,32),(3072*2,3072,64),(3072*2,3072,96),(3072*2,3072,128),
                    #(3072*3,3072,32),(3072*3,3072,64),(3072*3,3072,96),(3072*3,3072,128),
                    #(3072*4,3072,32),(3072*4,3072,64),(3072*4,3072,96),(3072*4,3072,128),): 
                    #(3072,3072,32+128*0),(3072,3072,64+128*0),(3072,3072,96+128*0),(3072,3072,128+128*0),
                    #(3072,3072,32+128*1),(3072,3072,64+128*1),(3072,3072,96+128*1),(3072,3072,128+128*1),
                    #(3072,3072,32+128*2),(3072,3072,64+128*2),(3072,3072,96+128*2),(3072,3072,128+128*2),
                    #(3072,3072,32+128*3),(3072,3072,64+128*3),(3072,3072,96+128*3),(3072,3072,128+128*3),):
        for op,  dimA,  dimB,  dimC in (
            ("nn", (K,C), (C,N), (K,N) ),  # fprop
            ("tn", (K,C), (K,N), (C,N) ),  # bprop
            ("nt", (K,N), (C,N), (K,C) )): # update
            # ("nn", (N,C), (C,K), (N,K) ),  # fprop
            # ("nt", (N,K), (C,K), (N,C) ),  # bprop
            # ("tn", (N,C), (N,K), (C,K) )): # update

            repeat = 500

            devA1 = ng.empty(dimA, dtype=dtype)
            devB1 = ng.empty(dimB, dtype=dtype)
            devC1 = ng.empty(dimC, dtype=dtype)

            # fill with uniform randoms from -1 to 1
            devA1[:] = 2 * (.5 - ng.rand())
            devB1[:] = 2 * (.5 - ng.rand())

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

            devC2 = ng.empty(dimC, dtype=np.float32)

            if op[0] == 't': devA1, devA2 = devA1.T, devA2.T
            if op[1] == 't': devB1, devB2 = devB1.T, devB2.T

            glops32x128 = 0
            glops128x32 = 0
            glops128x64 = 0

            if op != 'tn':
                glops32x128 = ng.dot(devA1, devB1, devC1, repeat=repeat, size='32x128')
            if op != 'nt':
                glops128x32 = ng.dot(devA1, devB1, devC1, repeat=repeat, size='128x32')
                glops128x64 = ng.dot(devA1, devB1, devC1, repeat=repeat, size='128x64')
            glops128x128 = ng.dot(devA1, devB1, devC1, repeat=repeat, size='128x128')

            glops = max(glops32x128, glops128x32, glops128x64, glops128x128)

            if glops32x128 == glops:
                fastest = '32x128'
            elif glops128x32 == glops:
                fastest = '128x32'
            elif glops128x64 == glops:
                fastest = '128x64'
            else:
                fastest = '128x128'

            glopsref = cublas_dot(devA2, devB2, devC2, repeat=repeat)

            partial1 = ng.empty((devC1.shape[0],1), dtype=np.float32)
            partial2 = partial1[0:1,0:1]

            diff = ng.max(abs(devC2 - devC1), partial=partial1, out=partial2).get()[0,0]
            mean = ng.mean(abs(devC2), partial=partial1, out=partial2).get()[0,0]

            flops_diff = glops - glopsref

            note = "**************" if flops_diff <= 0 else ""

            print("Faster: %.0f gflops Choice: %s Error: %.3f%%%s" %
                  (flops_diff, fastest, 100 * diff / mean, note))

        print("--------------------------------------------------------------------------------")


cublas.cublasDestroy(handle)

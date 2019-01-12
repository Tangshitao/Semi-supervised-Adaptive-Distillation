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

# Using just cublas compare N as the contiguous dimension verses the non-contiguous dimension.

import numpy as np
import pycuda.driver as drv
from nervanagpu import NervanaGPU
from pycuda.autoinit import context
from scikits.cuda import cublas

print(context.get_device().name())

ng = NervanaGPU(stochastic_round=False, bench=True)

handle = cublas.cublasCreate()

start, end = (drv.Event(), drv.Event())

def cublas_dot(op, A, B, C, repeat=1, warmup=False):

    lda = A.shape[0]
    ldb = B.shape[0]
    ldc = C.shape[0]

    m = C.shape[0]
    n = C.shape[1]
    k = A.shape[1] if op[0] == 'n' else A.shape[0]

    if warmup:
        for r in range(repeat):
            cublas.cublasSgemm(handle, op[0], op[1], m, n, k, 1.0, A.gpudata, lda, B.gpudata, ldb, 0.0, C.gpudata, ldc)

    start.record()
    
    # Swap A and B to map from C order to Fortran 
    for r in range(repeat):
        cublas.cublasSgemm(handle, op[0], op[1], m, n, k, 1.0, A.gpudata, lda, B.gpudata, ldb, 0.0, C.gpudata, ldc)

    end.record()
    end.synchronize()
    msecs = end.time_since(start) / repeat
    gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
    print("%7.3f msecs %4.0f gflops (%s: %d,%d,%d)" % (msecs,gflops,op,m,n,k))

    return msecs

# N non-contiguous:
# fprop(nn): KC   x CN   = KN
# bprop(tn): KC^T x KN   = CN
# updat(nt): KN   x CN^T = KC

# N contiguous:
# fprop(nt): NC   x KC^T = NK
# bprop(nn): NK   x KC   = NC
# updat(tn): NK^T x NC   = KC

repeat = 2000


for K, C, N in ((3072,3072,32),):

    total  = 0

    for op,  dimA,  dimB,  dimC in (
      ("nn", (K,C), (C,N), (K,N) ),   # fprop
      ("tn", (K,C), (K,N), (C,N) ),   # bprop
      ("nt", (K,N), (C,N), (K,C) ),): # update

        devA = ng.empty(dimA, dtype=np.float32)
        devB = ng.empty(dimB, dtype=np.float32)
        devC = ng.empty(dimC, dtype=np.float32)

        # fill with uniform randoms from -1 to 1
        devA[:] = 2 * (.5 - ng.rand())
        devB[:] = 2 * (.5 - ng.rand())

        total += cublas_dot(op, devA, devB, devC, repeat=repeat, warmup=True)

    print("N2 Total: ", total)
    total = 0

    for op,  dimA,  dimB,  dimC in (
      ("nt", (N,C), (K,C), (N,K) ),   # fprop
      ("nn", (N,K), (K,C), (N,C) ),   # bprop
      ("tn", (N,K), (N,C), (K,C) ),): # update

        devA = ng.empty(dimA, dtype=np.float32)
        devB = ng.empty(dimB, dtype=np.float32)
        devC = ng.empty(dimC, dtype=np.float32)

        # fill with uniform randoms from -1 to 1
        devA[:] = 2 * (.5 - ng.rand())
        devB[:] = 2 * (.5 - ng.rand())

        total += cublas_dot(op, devA, devB, devC, repeat=repeat)

    print("N1 Total: ", total)

    print("--------------------------------------------------------------------------------")

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

start, end = (drv.Event(), drv.Event())

handle = cublas.cublasCreate()

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

    if repeat > 1:
        start.record()
    
    # Swap A and B to map from C order to Fortran 
    for r in range(repeat):
        cublas.cublasSgemm(handle, opB, opA, n, m, k, alpha, B.gpudata, ldb, A.gpudata, lda, beta, C.gpudata, ldc)

    if repeat > 1:
        end.record()
        end.synchronize()
        msecs = end.time_since(start) / repeat
        gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
        print("%7.3f msecs %4.0f gflops (%s_%s   : %d,%d,%d)" %
              (msecs,gflops,"cublas",op,m,n,k))


np.set_printoptions(threshold=8193, linewidth=600, formatter={'float':lambda x: "% .0f" % x})

ng = NervanaGPU(stochastic_round=0, bench=0)

small_1  = (1,2,3,4,5,6,7,8,9,16,32,64,65,72,120,127,128,192)
medium_1 = (32,64,128,192,778,785,786,787,794)
big_1    = (32,64,128,1532,1535,1536,1537,1540,3073,4095)

small_2  = (8,16,32,64,72,96,120,128,192)
medium_2 = (32,64,128,192,256,786-32,786-16,786,786+16,786+32)
big_2    = (32,64,128,1536-80,1536-64,1536,1536+64,1536+80,3072,4096)

# small_1  = (31,33,63,65,127,129)
# medium_1 = (778,785,786,787,794)
# big_1    = (1535,1537,3073,4095)

# small_2  = (32,64,72,96,120,128,192)
# medium_2 = (786-32,786-16,786,786+16,786+32)
# big_2    = (1536-80,1536+80,3072,4096)


# sharedDim = (4096,4096)
# devA1s = ng.empty(sharedDim, dtype=np.float32)
# devB1s = ng.empty(sharedDim, dtype=np.float32)
# devC1s = ng.empty(sharedDim, dtype=np.float32)
# devA2s = ng.empty(sharedDim, dtype=np.float32)
# devB2s = ng.empty(sharedDim, dtype=np.float32)
# devC2s = ng.empty(sharedDim, dtype=np.float32)
# devPs  = ng.empty((sharedDim[0],1), dtype=np.float32)

implemented = {
    "nn" : set(("128x32","32x128","128x64","128x128")),
    "tn" : set(("128x32","128x64","128x128")),
    "nt" : set(("32x128","128x128")),
}

for dtype in (np.float16, np.float32, ): # np.float16
    
    maxerr = .005 if dtype is np.float32 else 0.7

    itemsize = np.dtype(dtype).itemsize

    print(dtype)
    
    for size in (small_1, small_2, medium_1, medium_2, big_1, big_2):
        print(size)

        for K in size:
            print("K:", K)
            for C in (size):
                print("C:", C)
                for N in (size):
                    for alpha, beta in ((1.0,0.0), (0.5,0.5)):

                        for op,  dimA,  dimB,  dimC in (
                          ("nn", (K,C), (C,N), (K,N) ),  # fprop
                          ("tn", (K,C), (K,N), (C,N) ),  # bprop
                          ("nt", (K,N), (C,N), (K,C) )): # update

                            try: 

                                devA1 = ng.empty(dimA, dtype=dtype)
                                devB1 = ng.empty(dimB, dtype=dtype)
                                devC1 = ng.empty(dimC, dtype=dtype)
                                # devA1 = devA1s.share(dimA, dtype=dtype)
                                # devB1 = devB1s.share(dimB, dtype=dtype)
                                # devC1 = devC1s.share(dimC, dtype=dtype)

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
                                    # devA2 = devA2s.share(dimA, dtype=np.float32)
                                    # devB2 = devB2s.share(dimB, dtype=np.float32)
                                    devA2[:] = devA1
                                    devB2[:] = devB1

                                devC2    = ng.empty(dimC, dtype=np.float32)
                                # devC2    = devC2s.share(dimC, dtype=np.float32)
                                devC2[:] = devC1

                                if op[0] == 't': devA1, devA2 = devA1.T, devA2.T
                                if op[1] == 't': devB1, devB2 = devB1.T, devB2.T

                                for tile in ("128x32","32x128",):
                                    if tile not in implemented[op]:
                                        continue
                                    try: 
                                        ng.dot(devA1, devB1, devC1, alpha=alpha, beta=beta, size=tile)
                                        #context.synchronize()

                                        cublas_dot(devA2, devB2, devC2, alpha=alpha, beta=beta)

                                        partial1 = ng.empty((devC1.shape[0],1), dtype=np.float32)
                                        #partial1 = devPs.share((devC1.shape[0],1), dtype=np.float32)
                                        partial2 = partial1[0:1,0:1]

                                        if ng.min(ng.finite(devC1), partial=partial1, out=partial2).get()[0,0] == 0.0:
                                            print("Error: NaN op: %s tile: %s KCN: (%d,%d,%d) ab: (%f,%f) dtype: %d" %
                                                  (op, tile, K,C,N, alpha,beta, itemsize))
                                            exit()

                                        diff = ng.max(abs(devC2 - devC1), partial=partial1, out=partial2).get()[0,0]
                                        mean = ng.mean(abs(devC2), partial=partial1, out=partial2).get()[0,0]
                                        pctErr = 100 * diff / mean

                                        if pctErr > maxerr:
                                            print("Error: %.3f%% diff: %.5f mean %.5f op: %s tile: %s KCN: (%d,%d,%d) ab: (%f,%f) dtype: %d" %
                                                  (pctErr, diff, mean, op, tile, K,C,N, alpha,beta, itemsize))
                                            print devC1.get()
                                            print devC2.get()
                                            exit()
                                    
                                    except drv.Error as e:
                                        print("op: %s tile: %s KCN: (%d,%d,%d) ab: (%f,%f) dtype: %d" %
                                              (op, tile, K,C,N, alpha,beta, itemsize))
                                        print(e)
                                        exit()

                            except drv.Error as e:
                                print("op: %s KCN: (%d,%d,%d) ab: (%f,%f) dtype: %d" %
                                      (op, K,C,N, alpha,beta, itemsize))
                                print(e)
                                exit()


cublas.cublasDestroy(handle)

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

import numpy         as np
import pycuda.driver as drv
from nervanagpu      import NervanaGPU
from pycuda.autoinit import context
from operator        import mul

print(context.get_device().name())

np.set_printoptions(threshold=8193, linewidth=600, formatter={'int':lambda x: "%10d" % x,'float':lambda x: "% .3f" % x})

dtype  = np.float32
cpu    = 1
repeat = 1

ng = NervanaGPU(stochastic_round=False, bench=True)

pool = ng.pool_layer(dtype,
    "max",
    32,         # N
    32,1,32,32, # C,D,H,W
    2,1,3,3,    # J,T,R,S
    0,0,0,0,    # padding
    2,1,2,2)    # strides

dimI = pool.dimI
dimO = pool.dimO

# colapse pooling dimensions into one
# this allows for easy cpu pooling in numpy
def slicable(dim, pad=0):
    dim0 = reduce(mul, dim[:-1], 1) + pad
    return (dim0, dim[-1])

# cpu input arrays
cpuI = np.random.uniform(0.0, 1.0, slicable(dimI,1)).astype(np.float16).astype(np.float32)

# zero pad the last row of cpu input for the sake of numpy
if pool.op == "max":
    cpuI[-1,:] = np.finfo(cpuI.dtype).min
else:
    cpuI[-1,:] = 0

# cpu output arrays
cpuO = np.empty(dimO, dtype=np.float32)
cpuB = np.zeros(slicable(dimI,1), dtype=np.float32)

# give gpu the input array without zero padding (not needed)
devI = ng.array(cpuI[:-1,:].reshape(dimI), dtype=dtype)
devO = ng.zeros(dimO, dtype=dtype)
devB = ng.empty(dimI, dtype=dtype)

ng.fprop_pool(pool, devI, devO, repeat=repeat)

ng.bprop_pool(pool, devI, devO, devB, repeat=repeat)

def pixel_indices(kj, mt, pr, qs):

    C       = pool.C
    J,T,R,S = pool.JTRS
    D,H,W = pool.DHW
    HW    = H*W
    DHW   = D*H*W
    imax  = C*D*H*W
    idx   = []

    for j in range(J):
        c  = kj + j
        ci = c*DHW
        cb = c >= 0 and c < C

        for t in range(T):
            z  = mt + t
            zi = ci + z*HW
            zb = cb and z >= 0 and z < D

            for r in range(R):
                y  = pr + r
                yi = zi + y*W
                yb = zb and y >= 0 and y < H

                for s in range(S):
                    x  = qs + s
                    if yb and x >= 0 and x < W:
                        xi = yi + x
                    else:
                        xi = imax  # out of bounds

                    idx.append(xi)
    return idx

# numpy pooling implementation
if cpu:

    op    = pool.op
    C     = pool.C
    K     = pool.K
    N     = pool.N
    M,P,Q = pool.MPQ
    pad_j, pad_d, pad_h, pad_w = pool.padding
    str_j, str_d, str_h, str_w = pool.strides

    for k in range(K):
        kj = k*str_j - pad_j

        for m in range(M):
            mt = m*str_d - pad_d

            for p in range(P):
                pr = p*str_h - pad_h

                for q in range(Q):
                    qs = q*str_w - pad_w

                    idx = pixel_indices(kj, mt, pr, qs)

                    if op == "max":

                        #set_trace()
                        cpuO[k,m,p,q,:] = np.max(cpuI[idx,:], axis=0)

                        b_idx = np.argmax(cpuI[idx,:], axis=0)

                        # There's probably a more elegant numpy way to do this..
                        for n in range(N):
                            cpuB[idx[b_idx[n]],n] += cpuO[k,m,p,q,n]

                    # bprop not implemented yet
                    elif op == "avg":
                        cpuO[k,m,p,q,:] = np.mean(cpuI[idx,:], axis=0)
                    elif op == "l2":
                        cpuO[k,m,p,q,:] = np.sqrt(np.sum(cpuI[idx,:]**2, axis=0))


    # drop zero padding
    cpuI = cpuI[:-1,:].reshape(dimI)
    cpuB = cpuB[:-1,:].reshape(dimI)

    #print(cpuI[1,0,:,:,0], "\n")
    #print(cpuO[1,0,:,:,0], "\n")
    # print(cpuB[0,0,:,:,0], "\n")

    devO = devO.get().astype(np.float32)
    devB = devB.get().astype(np.float32)

    #print(devO[1,0,:,:,0], "\n")
    # print(devB[0,0,:,:,0], "\n")

    difO = np.absolute(cpuO - devO)
    print("difO max: ", difO.max())

    difB = np.absolute(cpuB - devB)
    print("difB max: ", difB.max())

    # for c in range(C):
    #     for n in range(N):
    #         if difB[c,0,:,:,n].max() > 1:
    #             print(cpuI[c,0,:,:,n], "\n")
    #             print(cpuB[c,0,:,:,n], "\n")
    #             print(devB[c,0,:,:,n], "\n")
    #             print(difB[c,0,:,:,n], "\n")
    #             print(c, n)


    # print(difB)
    # print(np.argmax(difB) #[0,0,:,:,0], "\n")
    # print("difB max:", difB.max(), "\n")
    # for op, devA, cpuA in (
    #     ("fprop",devO,cpuO),
    #     ("bprop",devB,cpuB[:-1,:].reshape(dimI)),
    #     ("update",devU,cpuU.reshape(dimF))):

    #     if op not in ops: continue

    #     devA = devA.get().astype(np.float32)

    #     difA = np.absolute(cpuA - devA)
    #     print(op, "diff max:", difA.max(), "\n")

        # print(devA[0,0,::4,::4,0], "\n")
        # print(cpuA[0,0,::4,::4,0], "\n")
        # print(difA[0,0,::4,::4,0], "\n")
        # print(difA[1,0,::4,::4,0], "\n")
        # print(difA[2,0,::4,::4,0], "\n")

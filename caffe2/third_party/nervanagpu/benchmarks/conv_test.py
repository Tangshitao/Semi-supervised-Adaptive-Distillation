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
import sys

if sys.version_info >= (3, 0):
    from functools import reduce

print(context.get_device().name())

np.set_printoptions(threshold=8193, linewidth=600, formatter={'int':lambda x: "%10d" % x,'float':lambda x: "% .0f" % x})

ops  = set(("update",)) # "fprop","bprop","update"
ones = 0
cpu  = 0  # Set CPU to 1 to check against CPU
repeat = 1
dtype = np.float32

ng = NervanaGPU(stochastic_round=False, bench=True)

conv = ng.conv_layer(
    dtype,
    16,3,8,    # N,C,K
    1,64,64,   # D,H,W
    1,3,3,     # T,R,S
    0,1,1,     # padding
    1,1,1)     # strides


dimI = conv.dimI
dimF = conv.dimF
dimO = conv.dimO

# colapse outer dimensions into one and preserve inner dimension
# this allows for easy cpu convolution in numpy
def slicable(dim, pad=0):
    dim0 = reduce(mul, dim[:-1], 1) + pad
    return (dim0, dim[-1])

# cpu input arrays
if ones:
    cpuI = np.ones(slicable(dimI,1), dtype=np.float32)
    cpuF = np.ones(slicable(dimF),   dtype=np.float32)
    cpuE = np.ones(dimO,             dtype=np.float32)

    # for k in range(cpuF.shape[1]):
    #     cpuF[:,k] = np.arange(0,cpuF.shape[0], dtype=np.float32)

else:
    cpuI = np.random.uniform(-127.0, 127.0, slicable(dimI,1)).astype(np.float32) #.astype(np.uint8) .astype(np.int8)
    cpuF = np.random.uniform(0.0, 1.1, slicable(dimF)  ).astype(np.float32)
    cpuE = np.random.uniform(-1.01, 1.01, dimO            ).astype(np.float32)

# zero pad the last row of cpu input for the sake of numpy
cpuI[-1,:] = 0.0

# cpu output arrays
cpuO = np.zeros(dimO,             dtype=np.float32)
cpuB = np.zeros(slicable(dimI,1), dtype=np.float32)
cpuU = np.zeros(slicable(dimF),   dtype=np.float32)

# give gpu the input array without zero padding (not needed)
devI = ng.array(cpuI[:-1,:].reshape(dimI), dtype=dtype)
devF = ng.array(cpuF.reshape(dimF), dtype=dtype)
devE = ng.array(cpuE, dtype=dtype)

devO = devB = devU = 0

if "fprop"  in ops:
    devO = ng.empty(dimO, dtype=dtype)
    ng.fprop_conv(conv,  devI, devF, devO, alpha=1.0, repeat=repeat)

if "bprop"  in ops:
    devB = ng.empty(dimI, dtype=dtype)
    ng.bprop_conv(conv,  devF, devE, devB, alpha=1.0, repeat=repeat)

if "update" in ops:
    devU = ng.empty(dimF, dtype=dtype)
    ng.update_conv(conv, devI, devE, devU, alpha=1.0, repeat=repeat)


def pixel_indices(mt, pr, qs):

    T,R,S = conv.TRS
    D,H,W = conv.DHW
    C     = conv.C
    HW    = H*W
    DHW   = D*H*W
    imax  = C*DHW

    idx = []
    for c in range(C):
        ci = c*DHW

        for t in range(T):
            z  = mt + t
            zi = ci + z*HW
            zb = z >= 0 and z < D

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

# numpy convolution implementation

# devA = devO.get().astype(np.float32)
# print(devA[:,0,0,0,:])
# exit()

if cpu:

    D,H,W = conv.DHW
    T,R,S = conv.TRS
    M,P,Q = conv.MPQ

    pad_d, pad_h, pad_w = conv.padding
    str_d, str_h, str_w = conv.strides

    for m in range(M):
        mt = m*str_d - pad_d

        for p in range(P):
            pr = p*str_h - pad_h

            for q in range(Q):
                qs = q*str_w - pad_w

                idx = pixel_indices(mt, pr, qs)
                #print("mpq(%d,%d,%d)" % (m,p,q))
                #print(np.array(idx))

                if "fprop"  in ops:
                    cpuO[:,m,p,q,:] = np.dot( cpuF.T,      cpuI[idx,:]       )

                if "bprop"  in ops:
                    cpuB[idx,:]    += np.dot( cpuF,        cpuE[:,m,p,q,:]   )

                if "update" in ops:
                    cpuU           += np.dot( cpuI[idx,:], cpuE[:,m,p,q,:].T )

    # drop zero padding and reshape to match device tensors
    #cpuI = cpuI[:-1,:].reshape(dimI)
    #cpuF = cpuF.reshape(dimF)

    for op, devA, cpuA, w in (
        ("fprop",devO,cpuO, Q),
        ("bprop",devB,cpuB[:-1,:].reshape(dimI), W),
        ("update",devU,cpuU.reshape(dimF), S)):

        if op not in ops: continue

        devA = devA.get().astype(np.float32)
        difA = cpuA - devA
        maxval = abs(cpuA.max())
        maxdif = abs(difA.max())
        #difA = np.absolute(cpuA - devA)
        print(op, "diff max: %.4f %.4f %.4f\n" % (maxdif, maxval, maxdif / maxval))

        for c in range(difA.shape[0]):
            for n in range(difA.shape[4]):
                if abs(difA[c,0,:,:,n].max()) / maxval >= 0.01:

                    print(difA[c,0,:,:,n].shape)

                    pq = np.argmax(difA[c,0,:,:,n])

                    p = pq // w
                    q = pq % w

                    print(p, q)

                    print(cpuA[:,0,p,q,:], "\n")
                    print(devA[:,0,p,q,:], "\n")
                    print(difA[:,0,p,q,:])

                    # print(difA[:,0,y,x,:], "\n")
                    # print(cpuA[c,0,:,:,n], "\n")
                    # print(devA[c,0,:,:,n], "\n")
                    # print(difA[c,0,:,:,n], "\n")
                    # print(difA[c+1,0,:,:,n], "\n")
                    # print(difA[c,0,:,:,n+1], "\n")
                    # print(c, n, difA[c,0,:,:,n].max())
                    exit()

        # print(devA[0,0,::4,::4,0], "\n")
        # print(cpuA[0,0,::4,::4,0], "\n")
        # print(difA[0,0,::4,::4,0], "\n")
        # print(difA[1,0,::4,::4,0], "\n")
        # print(difA[2,0,::4,::4,0], "\n")



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

import ctypes
import libcudnn
import numpy         as np
import pycuda.driver as drv
from pycuda.autoinit import context
from nervanagpu      import NervanaGPU, GPUTensor
from math import sqrt
from time import sleep

print(context.get_device().name())

# Set dtype to float32 or float16
dtype  = np.float32
repeat = 10

start, end = (drv.Event(), drv.Event())

def start_bench():
    start.record()

def end_bench(op):
    end.record()
    end.synchronize()
    msecs  = end.time_since(start) / repeat
    gflops = conv.flops / (msecs * 1000000.0)
    print("%7.3f msecs %8.3f gflops (%s: %s)" % (msecs, gflops, op, conv))

ng = NervanaGPU(stochastic_round=False, bench=True)

# Create a cuDNN context
cudnn = libcudnn.cudnnCreate()

C_desc = libcudnn.cudnnCreateConvolutionDescriptor()
I_desc = libcudnn.cudnnCreateTensorDescriptor()
O_desc = libcudnn.cudnnCreateTensorDescriptor()
E_desc = libcudnn.cudnnCreateTensorDescriptor()
B_desc = libcudnn.cudnnCreateTensorDescriptor()
F_desc = libcudnn.cudnnCreateFilterDescriptor()
U_desc = libcudnn.cudnnCreateFilterDescriptor()

# Set some options and tensor dimensions
NCHW_fmt  = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
cu_dtype  = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
conv_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
fwd_pref  = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_NO_WORKSPACE']
# CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
# CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
# CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4

                # N    C   K  D    H   W  T  R  S   pad    str
for dims in (   ( 64,  3, 64, 1, 224,224, 1, 3, 3, 0,1,1, 1,1,1), # VGG
                ( 64, 64, 64, 1, 224,224, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64, 64,128, 1, 112,112, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64,128,128, 1, 112,112, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64,128,256, 1,  56, 56, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64,256,256, 1,  56, 56, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64,256,512, 1,  28, 28, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64,512,512, 1,  28, 28, 1, 3, 3, 0,1,1, 1,1,1),
                ( 64,512,512, 1,  14, 14, 1, 3, 3, 0,1,1, 1,1,1),

                (128,  3, 64, 1, 224,224, 1,11,11, 0,3,3, 1,4,4),  #Alexnet
                (128, 64,192, 1,  27, 27, 1, 5, 5, 0,2,2, 1,1,1),
                (128,192,384, 1,  13, 13, 1, 3, 3, 0,1,1, 1,1,1),
                (128,384,256, 1,  13, 13, 1, 3, 3, 0,1,1, 1,1,1),
                (128,256,256, 1,  13, 13, 1, 3, 3, 0,1,1, 1,1,1),):

    conv = ng.conv_layer(dtype, *dims)

    N,C,K = conv.NCK
    D,H,W = conv.DHW
    T,R,S = conv.TRS
    M,P,Q = conv.MPQ
    pad_d, pad_h, pad_w = conv.padding
    str_d, str_h, str_w = conv.strides
    alpha, beta = (1.0, 0.0)

    dimI = conv.dimI2
    dimF = conv.dimF2
    dimO = conv.dimO2

    print("cudnn:")

    cuI = ng.empty(dimI[::-1], dtype=np.float32)
    cuF = ng.empty(dimF[::-1], dtype=np.float32)
    cuE = ng.empty(dimO[::-1], dtype=np.float32)
    cuB = ng.empty(dimI[::-1], dtype=np.float32)
    cuU = ng.empty(dimF[::-1], dtype=np.float32)
    cuO = ng.empty(dimO[::-1], dtype=np.float32)
    cuI[:] = 2 * (.5 - ng.rand())
    cuF[:] = 2 * (.5 - ng.rand())
    cuE[:] = 2 * (.5 - ng.rand())

    #print(drv.mem_get_info())

    I_data = ctypes.c_void_p(int(cuI.gpudata))
    F_data = ctypes.c_void_p(int(cuF.gpudata))
    O_data = ctypes.c_void_p(int(cuO.gpudata))
    E_data = ctypes.c_void_p(int(cuE.gpudata))
    B_data = ctypes.c_void_p(int(cuB.gpudata))
    U_data = ctypes.c_void_p(int(cuU.gpudata))


    libcudnn.cudnnSetConvolution2dDescriptor(C_desc, pad_h, pad_w, str_h, str_w, 1, 1, conv_mode)
    libcudnn.cudnnSetTensor4dDescriptor(I_desc, NCHW_fmt, cu_dtype, N, C, H, W)
    libcudnn.cudnnSetTensor4dDescriptor(B_desc, NCHW_fmt, cu_dtype, N, C, H, W)
    libcudnn.cudnnSetTensor4dDescriptor(O_desc, NCHW_fmt, cu_dtype, N, K, P, Q)
    libcudnn.cudnnSetTensor4dDescriptor(E_desc, NCHW_fmt, cu_dtype, N, K, P, Q)
    libcudnn.cudnnSetFilter4dDescriptor(F_desc, cu_dtype, K, C, R, S)
    libcudnn.cudnnSetFilter4dDescriptor(U_desc, cu_dtype, K, C, R, S)

    algo    = libcudnn.cudnnGetConvolutionForwardAlgorithm(cudnn, I_desc, F_desc, C_desc, O_desc, fwd_pref, 0)
    ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn, I_desc, F_desc, C_desc, O_desc, algo)

    #print(algo.value, ws_size.value)

    ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
    ws_data = ctypes.c_void_p(int(ws_ptr))

    start_bench()
    for r in (range(repeat)):
        libcudnn.cudnnConvolutionForward(cudnn, alpha, I_desc, I_data, F_desc, F_data, C_desc, algo, ws_data, ws_size.value, beta, O_desc, O_data)
    end_bench("fprop")

    ws_ptr = None

    if C >= 64:
        start_bench()
        for r in (range(repeat)):
            libcudnn.cudnnConvolutionBackwardData(cudnn, alpha, F_desc, F_data, E_desc, E_data, C_desc, beta, B_desc, B_data)
        end_bench("bprop")

    start_bench()
    for r in (range(repeat)):
        libcudnn.cudnnConvolutionBackwardFilter(cudnn, alpha, I_desc, I_data, E_desc, E_data, C_desc, beta, U_desc, U_data)
    end_bench("updat")


    print("\nnervana_lib:")

    nlI = ng.empty(dimI, dtype=dtype)
    nlI[:] = cuI.T
    cuI = None

    nlF = ng.empty(dimF, dtype=dtype)
    nlF[:] = cuF.T
    cuF = None

    nlE = ng.empty(dimO, dtype=dtype)
    nlE[:] = cuE.T
    cuE = None

    nlB = ng.empty(dimI, dtype=dtype)
    nlU = ng.empty(dimF, dtype=np.float32)
    nlO = ng.empty(dimO, dtype=dtype)
    #print(drv.mem_get_info())

    ng.fprop_conv (conv, nlI, nlF, nlO, alpha=alpha, repeat=repeat)
    if C >= 64:
        ng.bprop_conv (conv, nlF, nlE, nlB, alpha=alpha, repeat=repeat)
    ng.update_conv(conv, nlI, nlE, nlU, alpha=alpha, repeat=repeat)

    nlI = nlF = nlE = None

    print("\ncudnn vs nervanaLib:")

    parO = ng.empty((N,1), dtype=np.float32)
    parB = ng.empty((N,1), dtype=np.float32)
    parU = ng.empty((K,1), dtype=np.float32)
    maxO = parO[0:1,0:1]
    maxB = parB[0:1,0:1]
    maxU = parU[0:1,0:1]

    maxo  = ng.max(abs(cuO - nlO.T), partial=parO, out=maxO).get()[0,0]
    if C >= 64:
        maxb  = ng.max(abs(cuB - nlB.T), partial=parB, out=maxB).get()[0,0]
    maxu  = ng.max(abs(cuU - nlU.T), partial=parU, out=maxU).get()[0,0]

    meano = ng.mean(abs(cuO), partial=parO, out=maxO).get()[0,0]
    if C >= 64:
        meanb = ng.mean(abs(cuB), partial=parB, out=maxB).get()[0,0]
    meanu = ng.mean(abs(cuU), partial=parU, out=maxU).get()[0,0]

    print("        maxerr   mean   pct")
    print("fprop: %7.5f %6.2f %5.3f" % (maxo, meano, 100*maxo/meano))
    if C >= 64:
        print("bprop: %7.5f %6.2f %5.3f" % (maxb, meanb, 100*maxb/meanb))
    print("updat: %7.5f %6.2f %5.3f" % (maxu, meanu, 100*maxu/meanu))

    # free up memory from this layer before proceeding
    cuB  = cuU  = cuO  = None
    nlB  = nlU  = nlO  = None
    parO = parB = parU = maxO = maxB = maxU = None


libcudnn.cudnnDestroyTensorDescriptor(I_desc)
libcudnn.cudnnDestroyTensorDescriptor(O_desc)
libcudnn.cudnnDestroyFilterDescriptor(F_desc)
libcudnn.cudnnDestroyTensorDescriptor(E_desc)
libcudnn.cudnnDestroyTensorDescriptor(B_desc)
libcudnn.cudnnDestroyFilterDescriptor(U_desc)
libcudnn.cudnnDestroyConvolutionDescriptor(C_desc)

libcudnn.cudnnDestroy(cudnn)

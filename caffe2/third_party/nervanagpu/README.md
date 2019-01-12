# NervanaGPU library

## Deprecated

NervanaGPU is now maintained as a backend inside of the [neon](https://github.com/NervanaSystems/neon) project. This repository remains for reference only and is no longer updated.

## Introduction

**nervanagpu** is a Python module for deep learning. It includes,

- matrix-multiply (GEMM), convolution, and pooling kernels optimized using a custom [assembler](https://github.com/NervanaSystems/maxas),
- element-wise and broadcast operations that automatically compound into efficient kernels,
- a simple but powerful array class, leveraging and with code partially borrowed from pycuda<sup>[1](#refs)</sup>,
- layer classes for building networks for benchmarking,
- full assembler source to encourage contributions from the community.

#### Design goals

**nervanagpu** grew out of a tool Nervana uses for internal hardware efforts. It's been repackaged for use by the community. The goals for **nervanagpu** are to provide,

- near **theoretical peak performance**,
- numpy functionality for **ease-of-use**,
- convolution kernel features and arguments identical to cuDNN<sup>[2](#refs)</sup>,
- integration into [neon](https://github.com/NervanaSystems/neon), Nervana's full-featured deep learning library,
- a tool for algorithmic explorations using alternative numerical formats,
- a seemless transition path to Nervana hardware,
- ease of integration into other deep learning frameworks.

Only NVIDIA Maxwell and future architectures are supported. Older architectures are not well-suited for assembler level optimizations used here.

#### Numerical formats

Supported numerical formats currently include,

- **fp32**: standard 32-bit floating point,
- **fp16**: 16-bit floating point memory format with underlying operations in 32 bits.
- **int8** and **uint8**: in elementwise and as input to the first convolutional layer.

with more to come (eg. like [this](https://github.com/NervanaSystems/nervana-lib-gpu-performance-preview)).

#### Extra features

Our kernels have some additional useful features:

- 3D convolutions and 4D pooling (including output feature map dim)
- optional ReLu is builtin to GEMM and convolution operations,
- stochastic rounding support for **fp16**<sup>[3](#refs)</sup>,
- instrumented to return statistics useful for avoiding numerical issues (coming soon),
- support for matrix sizes common in deep learning, significantly out performing cuBLAS

Small optimizations like these can result in significant speed and performance improvements.

## Prerequisites

PyCUDA release 2015.1 is necessary in order to run **nervanagpu**.

## Usage

**nervanagpu** includes a factory class `NervanaGPU` and a numpy-like array class `GPUTensor`. Memory layout for tensors and gemm ops is **row-ordered**. Below are examples on how they are used.

### Matrix multiplication example

Here is full example of doing a basic GEMM operation using 16-bit float:

```python
import numpy as np
import pycuda.autoinit
from nervanagpu import NervanaGPU

# initialize factory class
ng = NervanaGPU(stochastic_round=False)

m, n, k  = 10, 20, 10
dtype = np.float16

# create matrices on host
cpuA = np.random.randn(k,m)
cpuB = np.random.randn(k,n)

# transfer to device
devA = ng.array(cpuA, dtype=dtype)
devB = ng.array(cpuB, dtype=dtype)
devC = ng.empty((m,n), dtype=dtype)

# do GEMM operation
ng.dot(devA.T, devB, devC, relu=False)

# get from device
cpuC = devC.get()
```

### Element-wise operations

**nervanagpu** compiles tensor arithmetic expressions into efficient CUDA kernels which are lazily evaluated upon assignment. For example, computing variance along an axis consists of a set of element-wise, reduction and broadcast operations that compiles to a single kernel (this code is also provided by the ng.var operator):

```python
# import and initialize NervanaGPU, transfer matrix from cpu to dev as above

devC[:] = ng.mean(ng.square(devA - ng.mean(devA, axis=1)), axis=1)

```

Batch normalization can be done by computing mean and variance across the batch (n) dimension and automatically taking advantage of broadcasting to subtract and divide the original data.

```python
# import and initialize NervanaGPU as above

eps  = .001 # for avoiding division by zero
A    = ng.empty((128, 32), dtype=np.float16)
A[:] = ng.rand() # generate uniform random on device between 0 and 1

# Normalize batch data by batch mean and variance,
A[:] = (A - ng.mean(A, axis=1)) / ng.sqrt(ng.var(A, axis=1) + eps)

```
The last expression above is automatically collapsed into a single gpu kernel. There are two mean(A,axis=1) operations embedded in that expression (one in the numerator and one inside the variance operation).  One of them is automatically optimized away, leading to the most efficient kernel possible.

## Building

**nervanagpu** comes with full assembler code for kernels. To build the kernels, install [**maxas**](https://github.com/NervanaSystems/maxas), Nervana's assembler for NVIDIA Maxwell. The module can then be built by running:

    make kernels      # build the kernels
    make python       # build python bindings
    make test         # run nosetests
    make doc          # build sphinx docs

A simple `make` will perform the first two steps.

Documentation and tests are currently sparse. Please contribute.

## Performance

**nervanagpu** comes with a set of benchmark scripts under `nervanagpu/benchmarks`. Also included are scripts to validate kernel results against cuBLAS and cuDNN.

Here is a sample run of `benchmarks/convnet-benchmarks.py` using the networks listed on Soumith Chintala's [benchmarking page](https://github.com/soumith/convnet-benchmarks).  Run on a single TitanX with default clocks and power limit:

    ---------------------------------------------
    Alexnet (dtype=float16, N=128) Results:
    ---------------------------------------------
    Avg(10) fprop:   28.341 msecs 6290.357 gflops
    Avg(10) bprop:   60.768 msecs 5867.390 gflops
    Avg(10) total:   89.109 msecs 6001.914 gflops
    ---------------------------------------------
    Alexnet (dtype=float32, N=128) Results:
    ---------------------------------------------
    Avg(10) fprop:   29.332 msecs 6077.931 gflops
    Avg(10) bprop:   66.216 msecs 5384.625 gflops
    Avg(10) total:   95.548 msecs 5597.458 gflops

    ---------------------------------------------
    Overfeat (dtype=float16, N=128) Results:
    ---------------------------------------------
    Avg(10) fprop:  107.401 msecs 6667.444 gflops
    Avg(10) bprop:  236.339 msecs 6059.859 gflops
    Avg(10) total:  343.741 msecs 6249.698 gflops
    ---------------------------------------------
    Overfeat (dtype=float32, N=128) Results:
    ---------------------------------------------
    Avg(10) fprop:  117.611 msecs 6088.632 gflops
    Avg(10) bprop:  268.379 msecs 5336.422 gflops
    Avg(10) total:  385.990 msecs 5565.621 gflops

    ---------------------------------------------
    VGG (dtype=float16, N=64) Results:
    ---------------------------------------------
    Avg(10) fprop:  153.939 msecs 6298.629 gflops
    Avg(10) bprop:  337.443 msecs 5746.783 gflops
    Avg(10) total:  491.382 msecs 5919.665 gflops
    ---------------------------------------------
    VGG (dtype=float32, N=64) Results:
    ---------------------------------------------
    Avg(10) fprop:  165.948 msecs 5842.816 gflops
    Avg(10) bprop:  389.139 msecs 4983.337 gflops
    Avg(10) total:  555.087 msecs 5240.286 gflops

    ---------------------------------------------
    VGG_E (dtype=float16, N=64) Results:
    ---------------------------------------------
    Avg(10) fprop:  396.107 msecs 6332.999 gflops
    Avg(10) bprop:  854.090 msecs 5874.195 gflops
    Avg(10) total: 1250.197 msecs 6019.560 gflops
    ---------------------------------------------
    VGG_E (dtype=float32, N=64) Results:
    ---------------------------------------------
    Avg(10) fprop:  435.622 msecs 5758.533 gflops
    Avg(10) bprop:  971.928 msecs 5162.000 gflops
    Avg(10) total: 1407.550 msecs 5346.621 gflops


#### Acknowledgements

Thanks to Erich Elsen and Bryan Catanzaro of Baidu, Matthieu Courbariaux and Frédéric Bastien of the Bengio lab, Vincent Vanhoucke of Google, and Soumith Chintala of Facebook for feedback on early versions of this library. Thanks to Andreas Klöckner for help with interfacing to his PyCUDA library.  We'd also like to thank NVIDIA for generously providing us with several TitanXs for benchmarking.


#### References <a name="refs"></a>

1. Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov, Ahmed Fasih.
[*PyCUDA and PyOpenCL: A scripting-based approach to GPU run-time code generation*](http://arxiv.org/abs/0911.3456)
Parallel Computing, Volume 38, Issue 3, March 2012, Pages 157-174.

2. Chetlur, Sharan, Cliff Woolley, Philippe Vandermersch, Jonathan
Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer.
[*cuDNN: Efficient primitives for deep learning.*](http://arxiv.org/abs/1410.0759)
arXiv preprint arXiv:1410.0759 (2014).

3. Gupta, Suyog, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. [*Deep Learning with Limited Numerical Precision.*](http://arxiv.org/abs/1502.02551) arXiv preprint arXiv:1502.02551 (2015).


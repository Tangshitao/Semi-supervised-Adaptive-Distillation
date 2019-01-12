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

import os
import sys
import numpy as np
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from struct import unpack_from
from pytools import memoize, memoize_method
from .float_ew import call_compound_kernel, fp32_convert, _get_compensated_sum_kernel, _get_fast_ew_dims, _get_transpose_kernel, _get_shuffle_kernel
from .layers import DataLayer, FullLayer, ConvLayer, DeconvLayer, PoolLayer, _get_sm_count

if sys.version_info >= (3, 0):
    from functools import reduce

_none_slice = slice(None,None,None)

class GPUTensor(object):

    def __init__(self, backend, shape,
                dtype      = np.float16,
                allocator  = drv.mem_alloc,
                base       = None,
                gpudata    = None,
                strides    = None,
                take_array = None,
                is_trans   = False,
                name       = None,
                rounding   = 0):

        # supported dtypes
        assert dtype in (np.float16, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32)

        dtype = np.dtype(dtype)

        try:
            size = 1
            for dim in shape:
                size *= dim
        except TypeError:
            assert isinstance(shape, (int, long, np.integer))
            size  = shape
            shape = (shape,)

        if isinstance(size, np.integer):
            size = np.asscalar(size)

        # only support C ordering for now.
        if strides is None:
            self.strides = _contiguous_strides(shape)
        else:
            self.strides = tuple(strides)

        self.backend     = backend
        self.base        = base
        self.shape       = shape
        self.size        = size
        self.dtype       = dtype
        self.nbytes      = dtype.itemsize * size
        self.allocator   = allocator
        self.take_array  = take_array
        self.is_trans    = is_trans
        self.name        = name
        self.rounding    = rounding
        self.kahan_count = 0
        self.kahan_reset = 0

        if gpudata is None:
            if size:
                #print(drv.mem_get_info())
                self.gpudata = allocator(self.nbytes)
            else:
                self.gpudata = None

            assert base is None
        else:
            self.gpudata = gpudata

    def __str__(self):
        return ("Array(0x%x) name:%s dtype:%s shape:%s strides:%s "
                " is_trans:%s" % (self.gpudata, self.name, self.dtype,
                self.shape, self.strides, self.is_trans))

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return int(self.gpudata)

    @property
    def ptr(self):
        return self.gpudata.__int__()

    def __len__(self):
        """Return the size of the leading dimension of self."""
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

    @property
    @memoize_method
    def is_contiguous(self):
        return not self.take_array and self.strides == _contiguous_strides(self.shape)

    def set(self, ary):
        """
        copy host array to device.
        Arguments:
            ary: host array, needs to be contiguous
        Returns:
            self
        """
        stream = self.backend.stream
        assert ary.size == self.size
        assert self.is_contiguous, "Array in set() must be contiguous"
        if ary.dtype is not self.dtype:
            ary = ary.astype(self.dtype)
        assert ary.strides == tuple(self.dtype.itemsize*s for s in self.strides)

        drv.memcpy_htod_async(self.gpudata, ary, stream)

        return self

    def get(self, stream=None):
        """
        copy device array to host.
        Returns:
            the host numpy array
        """
        assert self.is_contiguous, "Array in get() must be contiguous"
        ary = np.empty(self.shape, self.dtype)
        drv.memcpy_dtoh_async(ary, self.gpudata, stream)
        return ary

    def asnumpyarray(self):
        """
        asnumpyarray is an alias of get(), needed for MOP compatibility
        """
        return self.get()

    def asbuffer(self):
        """
        asbuffer returns buffer interface to gpu data
        """
        return self.gpudata.as_buffer(self.nbytes)

    def take(self, indices, axis, out=None):
        if axis == 1:
            view = self.__getitem__((_none_slice, indices))
        else:
            view = self.__getitem__((indices, _none_slice))

        if out:
            return out._assign(view)
        return view

    def __getitem__(self, index):
        """
        return a sliced view of an array
        """
        if not isinstance(index, tuple):
            # speed up common case of [:]
            if index == _none_slice:
                return self
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False
        take_array = None

        index_axis = 0
        array_axis = 0
        while index_axis < len(index):
            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            # Standard slicing (start:stop:step)
            if isinstance(index_entry, slice):
                start, stop, idx_stride = index_entry.indices(self.shape[array_axis])

                array_stride = self.strides[array_axis]

                # def ceil_div(x, y): return -(-x // y)
                new_shape.append( -((start-stop) // idx_stride) )
                new_strides.append(idx_stride*array_stride)
                new_offset += array_stride*start*self.dtype.itemsize

                index_axis += 1
                array_axis += 1

            # Fancy indexing
            elif isinstance(index_entry, (GPUTensor, np.ndarray, list, tuple)):

                if isinstance(index_entry, (list, tuple)):
                    index_entry = np.array(index_entry, dtype=np.int32)

                if isinstance(index_entry, np.ndarray):
                    index_entry = self.__class__(self.backend, index_entry.shape, dtype=np.int32).set(index_entry)

                size = max(index_entry.shape)
                if size != index_entry.size:
                    raise IndexError("Fancy indexing only currently supported for dim > 1 in a single dimension.")

                if take_array is not None:
                    raise IndexError("Fancy indexing only currently supported for one axis at a time.")

                if index_entry.dtype.type is not np.int32:
                    #TODO: this should now work for all int types, but need to test
                    raise IndexError("Fancy indexing only currently supported with int32 types.")

                take_array = (index_entry, array_axis)

                new_shape.append(size)
                new_strides.append(self.strides[array_axis])

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError("subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis]*index_entry*self.dtype.itemsize

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError("more than one ellipsis not allowed in index")
                seen_ellipsis = True

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return self.__class__(
                backend    = self.backend,
                shape      = tuple(new_shape),
                dtype      = self.dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = int(self.gpudata)+new_offset,
                strides    = new_strides,
                take_array = take_array,
                name       = self.name,
                rounding   = self.rounding)

    def _assign(self, value):

        stream = self.backend.stream
        if isinstance(value, (int, float)):

            # if we have a contiguous array, then use the speedy driver kernel
            if self.is_contiguous:

                value = self.dtype.type(value)

                if self.dtype.itemsize == 1:
                    drv.memset_d8_async( self.gpudata, unpack_from('B', value)[0], self.size, stream)
                elif self.dtype.itemsize == 2:
                    drv.memset_d16_async(self.gpudata, unpack_from('H', value)[0], self.size, stream)
                else:
                    drv.memset_d32_async(self.gpudata, unpack_from('I', value)[0], self.size, stream)

            # otherwise use our copy kerel
            else:
                OpTreeNode.build("assign", self, value)

        elif isinstance(value, GPUTensor):
            # TODO: add an is_binary_compat like function
            if self.is_contiguous and value.is_contiguous and self.dtype == value.dtype:
                drv.memcpy_dtod_async(self.gpudata, value.gpudata, self.nbytes, stream)
            else:
                OpTreeNode.build("assign", self, value)

        # collapse and execute an op tree as a kernel
        elif isinstance(value, OpTreeNode):
            OpTreeNode.build("assign", self, value)

        # assign to numpy array (same as set())
        elif isinstance(value, np.ndarray):
            self.set(value)

        else:
            raise TypeError("Invalid type for assignment: %s" % type(value))

        return self

    def __setitem__(self, index, value):

        self.__getitem__(index)._assign(value)

    def fill(self, value):
        return self._assign(value)


    def copy(self, a):
        return self._assign(a)

    def copy_from(self, a):
        """ alias of copy"""
        return self.set(a)

    def reshape(self, *shape):
        """
        return a reshaped view
        """
        if isinstance(shape[0], (tuple,list)):
            shape = tuple(shape[0])

        if shape == self.shape:
            return self

        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        if not self.is_contiguous:
            raise TypeError("reshaping of non-contigous "
                            "arrays is not yet supported")

        return self.__class__(
                backend    = self.backend,
                shape      = shape,
                dtype      = self.dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = self.gpudata,
                strides    = _contiguous_strides(shape),
                name       = self.name,
                rounding   = self.rounding)

    def share(self, shape, dtype=None, name=None):
        """
        return a view: ary, where ary.size <= self.size
        Allows easy sharing of temporary memory
        """
        size = reduce(lambda x, y: x * y, shape, 1)
        if size > self.size:
            raise ValueError("total size of new array must <= size of parent")

        if not self.is_contiguous:
            raise TypeError("sharing of non-contigous "
                            "arrays is not yet supported")

        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        return self.__class__(
                backend    = self.backend,
                shape      = shape,
                dtype      = dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = self.gpudata,
                strides    = _contiguous_strides(shape),
                name       = name,
                rounding   = self.rounding)

    @property
    def T(self):
        """
        return a transposed view
        """
        if len(self.shape) <= 2:
            shape   = self.shape[::-1]
            strides = self.strides[::-1]
        else:
            # support for batched dot.
            # perserve outer dimension but reverse inner dims
            shape   = list(self.shape[::-1])
            strides = list(self.strides[::-1])
            shape   = tuple(shape[-1:]   + shape[:-1])
            strides = tuple(strides[-1:] + strides[:-1])

        return self.__class__(
                backend    = self.backend,
                shape      = shape,
                dtype      = self.dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = self.gpudata,
                strides    = strides,
                is_trans   = not self.is_trans,
                name       = self.name,
                rounding   = self.rounding)

    def transpose(self, out=None):
        """
        Return a transposed view of the data.  Alias of .T property needed for
        MOP compatibility.
        """
        if out:
            return OpTreeNode.build("assign", out, self.T)
        return self.T

    def __add__      (self, other): return OpTreeNode.build("add", self, other)
    def __sub__      (self, other): return OpTreeNode.build("sub", self, other)
    def __mul__      (self, other): return OpTreeNode.build("mul", self, other)
    def __div__      (self, other): return OpTreeNode.build("div", self, other)
    def __truediv__  (self, other): return OpTreeNode.build("div", self, other)
    def __pow__      (self, other): return OpTreeNode.build("pow", self, other)
    def __radd__     (self, other): return OpTreeNode.build("add", other, self)
    def __rsub__     (self, other): return OpTreeNode.build("sub", other, self)
    def __rmul__     (self, other): return OpTreeNode.build("mul", other, self)
    def __rdiv__     (self, other): return OpTreeNode.build("div", other, self)
    def __rtruediv__ (self, other): return OpTreeNode.build("div", other, self)
    def __rpow__     (self, other): return OpTreeNode.build("pow", other, self)
    def __eq__       (self, other): return OpTreeNode.build("eq",  self, other)
    def __ne__       (self, other): return OpTreeNode.build("ne",  self, other)
    def __lt__       (self, other): return OpTreeNode.build("lt",  self, other)
    def __le__       (self, other): return OpTreeNode.build("le",  self, other)
    def __gt__       (self, other): return OpTreeNode.build("gt",  self, other)
    def __ge__       (self, other): return OpTreeNode.build("ge",  self, other)
    def __abs__      (self):        return OpTreeNode.build("abs", self,  None)
    def __neg__      (self):        return OpTreeNode.build("neg", self,  None)

    def __iadd__     (self, other): return OpTreeNode.build("add", self, other, out=self)
    def __isub__     (self, other): return OpTreeNode.build("sub", self, other, out=self)
    def __imul__     (self, other): return OpTreeNode.build("mul", self, other, out=self)
    def __idiv__     (self, other): return OpTreeNode.build("div", self, other, out=self)
    def __itruediv__ (self, other): return OpTreeNode.build("div", self, other, out=self)
    def __ipow__     (self, other): return OpTreeNode.build("pow", self, other, out=self)

    #def __nonzero__  (self): raise ValueError("The truth value of an array with more than one element is ambiguous.")


class NervanaGPU(object):

    def __init__(self, stochastic_round=False, bench=False,
                 cubin_path=os.path.join("kernels", "cubin"),
                 scratch_size=9*1024*1024, default_dtype=np.float16):
        """
        NervanaGPU: the primary interface class and factory for GPUTensors

        stochastic_round: set to desired number of mantissa bits to stochasicaly round to
                          set to zero to disable stochastic rouding.
        bench: set to 1 to print out performance data for most kernel calls
        """
        if stochastic_round:
            if stochastic_round is True:
                stochastic_round = 10
        else:
            stochastic_round = 0

        self.scratch_size = scratch_size
        self.round_mode = stochastic_round
        self.cubin_path = os.path.join(os.path.dirname(__file__), cubin_path)
        self.bench = bench
        self.stream = None
        self.default_dtype = default_dtype

    def empty(self, shape, dtype=None, name=None, allocator=drv.mem_alloc):
        """
        allocate the space for a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype, allocator=allocator,
                          name=name, rounding=self.round_mode)

    def array(self, ary, dtype=None, name=None, allocator=drv.mem_alloc):
        """
        converts a numpy array to a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, ary.shape, dtype, allocator=allocator,
                          name=name, rounding=self.round_mode).set(ary)

    def zeros(self, shape, dtype=None, name=None, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 0's.
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype, allocator=allocator,
                          name=name, rounding=self.round_mode)._assign(0)

    def ones(self, shape, dtype=None, name=None, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 1's.
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype, allocator,
                          name=name, rounding=self.round_mode)._assign(1)

    def empty_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(self, other_ary.shape, other_ary.dtype, other_ary.allocator,
                          name=name, rounding=self.round_mode)

    def zeros_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(self, other_ary.shape, other_ary.dtype, other_ary.allocator,
                          name=name, rounding=self.round_mode)._assign(0)

    def conv_layer(self, dtype,
            N, C, K,
            D=1, H=1, W=1,
            T=1, R=1, S=1,
            pad_d=0, pad_h=0, pad_w=0,
            str_d=1, str_h=1, str_w=1,
            grid_P=0, grid_Q=0, update_size=None):
        """
        Create a new ConvLayer parameter object.
        This then is passed as an argument to all the convolution operations.

        N: Number of images in mini-batch
        C: Number of input feature maps
        K: Number of output feature maps

        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        T: Depth  of filter kernel
        R: Height of filter kernel
        S: Width  of filter kernel

        padding: amount of zero-padding around the given edge
        strides: factor to step the filters by in a given direction

        grid_P, grid_Q: For the update operation define the size of the grid
        to distribute the work accross SMs.  The smaller the grid, the deeper the
        MM and hence more accumulation is done in fp32.  The bigger the grid,
        the more the work can be evenly spanned accross the SMs, at the cost of
        needing more fp16 accumuation operations and increased error.

        Set to 1,1 for full fp32 accuracy
        Set to P,Q for maximal distribution of work acrross SMs
        Set to 0,0 for automactially calculated optimal balance (recommened).

        Tweaking these params can have a large impact on performance as the
        L2 cache utilization is greatly effected by them.

        update_size: override kernel size selection for update.
            "C128_K64"
            "C128_K128"

        dtype: need to know dtype to setup proper kernels and params.

        Maximum utilization is achieved when N, K and C*R*S*T is
        a multiple of 64
        """
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w, grid_P, grid_Q, update_size)

    def deconv_layer(self, dtype,
            N, C, K,
            P, Q,
            R=1, S=1,
            pad_d=0, pad_h=0, pad_w=0,
            str_d=1, str_h=1, str_w=1,
            grid_P=0, grid_Q=0, update_size=None):
        """
        Create a new DeconvLayer parameter object.
        This then is passed as an argument to all the convolution operations.

        N: Number of images in mini-batch
        C: Number of output feature maps
        K: Number of input feature maps

        P: Height of input
        Q: Width of input

        T: Depth  of filter kernel
        R: Height of filter kernel
        S: Width  of filter kernel

        padding: amount of zero-padding around the given edge
        strides: factor to step the filters by in a given direction

        grid_P, grid_Q: For the update operation define the size of the grid
        to distribute the work accross SMs.  The smaller the grid, the deeper the
        MM and hence more accumulation is done in fp32.  The bigger the grid,
        the more the work can be evenly spanned accross the SMs, at the cost of
        needing more fp16 accumuation operations and increased error.

        Set to 1,1 for full fp32 accuracy
        Set to P,Q for maximal distribution of work acrross SMs
        Set to 0,0 for automactially calculated optimal balance (recommened).

        Tweaking these params can have a large impact on performance as the
        L2 cache utilization is greatly effected by them.

        update_size: override kernel size selection for update.
            "C128_K64"
            "C128_K128"

        dtype: need to know dtype to setup proper kernels and params.

        Maximum utilization is achieved when N, K and C*R*S*T is
        a multiple of 64
        """
        return DeconvLayer(self, dtype, N, C, K, P, Q, R, S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w, grid_P, grid_Q, update_size)


    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size

        return self._execute_conv(
            layer, "fprop", layer.fprop_size,
            layer.fprop_grid, layer.fprop_block, layer.fprop_args, layer.fprop_lut_size,
            I, F, O, alpha, relu, False, repeat)

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, repeat=1):
        assert layer.sizeF == F.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size
        return self._execute_conv(
            layer, "bprop", layer.bprop_size,
            layer.bprop_grid, layer.bprop_block, layer.bprop_args, layer.bprop_lut_size,
            E, F, grad_I, alpha, False, layer.bprop_zero, repeat)

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == grad_F.size

        return self._execute_conv(
            layer, "updat", layer.updat_size,
            layer.updat_grid, layer.updat_block, layer.update_args, 0,
            I, E, grad_F, alpha, False, True, repeat)

    def _execute_conv(self, layer, op, size, grid, block, args, shared, A, B, C, alpha, relu, zero, repeat):

        assert A.dtype == B.dtype

        clss  = "hconv" if A.dtype.type is np.float16 else "sconv"

        flags = 0
        if relu: flags |= 2

        B_gpudata      = B.gpudata
        C_gpudata      = C.gpudata
        shuffle_kernel = None
        convert_data   = False

        if   op == "bprop":
            assert B.size <= self.scratch_size
            B_gpudata      = _get_scratch_data(self.scratch_size)
            if zero:
                shuffle_kernel = _get_transpose_kernel(B.dtype.str[1:])
            else:
                shuffle_kernel = _get_shuffle_kernel(B.dtype.str[1:])
            shuffle_args   = [ layer.shuffle_grid, layer.shuffle_block, self.stream,
                               B_gpudata, B.gpudata ] + layer.shuffle_args

        if op == "updat" and C.dtype.type is not np.float32:
            assert C.size <= self.scratch_size
            C_gpudata      = _get_scratch_data(self.scratch_size)
            convert_data   = True

        kernel = _get_conv_kernel(self.cubin_path, clss, op, size)
        params = [ grid, block, self.stream,
                   C_gpudata, A.gpudata, B_gpudata,
                   alpha, flags ] + args

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params, shared_size=shared)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(stream=self.stream)

        for r in range(repeat):
            if zero:
                drv.memset_d8_async(C_gpudata, 0, C.nbytes, self.stream)

            if shuffle_kernel:
                shuffle_kernel.prepared_async_call(*shuffle_args)

            kernel.prepared_async_call(*params, shared_size=shared)

            if convert_data:
                fp32_convert(C_gpudata, C)

        if self.bench or repeat > 1:
            end.record(stream=self.stream)
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            gflops = layer.flops / (msecs * 1000000.0)
            print("%7.3f msecs %8.3f gflops %6.0f (%s: %s) size:%s grid:%s" %
                  (msecs, gflops, layer.flops/1000000.0, op, layer, size, grid))
            return msecs, gflops
        return 0,0

    def pool_layer(self, dtype,
            op, N, C,
            D=1, H=1, W=1,
            J=1, T=1, R=1, S=1,
            pad_j=0, pad_d=0, pad_h=0, pad_w=0,
            str_j=None, str_d=None, str_h=None, str_w=None):
        """
        Create a new PoolLayer parameter object.
        This then is passed as an argument to all pooling kernels.

        op: max, avg, l2 pooling
        N: Number of images in mini-batch

        C: Number of input feature maps
        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        J: Size of feature map pooling window (maxout n_pieces)
        T: Depth  of pooling window
        R: Height of pooling window
        S: Width  of pooling window

        padding: amount of zero-padding around the given image or feature map edge
        strides: factor to step the window by in a given direction (overlap allowed)

        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.
        """
        # default to non-overlapping
        if str_j is None: str_j = J
        if str_d is None: str_d = T
        if str_h is None: str_h = R
        if str_w is None: str_w = S

        return PoolLayer(self, dtype, op, N, C, D, H, W, J, T, R, S,
            pad_j, pad_d, pad_h, pad_w, str_j, str_d, str_h, str_w)

    def fprop_pool(self, layer, I, O, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeO == O.size

        return self._execute_pool(layer, I, O, None, 0, repeat)

    def bprop_pool(self, layer, I, E, grad_I, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size
        assert I.dtype     == grad_I.dtype

        return self._execute_pool(layer, I, E, grad_I, 1, repeat)

    def _execute_pool(self, layer, I, O, B, mode, repeat):

        assert I.dtype == O.dtype

        clss = "hpool" if I.dtype.type is np.float16 else "spool"

        b_data = 0 if B is None else B.gpudata

        kernel = _get_pool_kernel(self.cubin_path, clss, "max")
        params = [layer.grid, layer.block, self.stream, I.gpudata, O.gpudata, b_data, mode]
        params.extend(layer.kernel_args)

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params, shared_size=layer.lut_size)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            if mode: B.fill(0)
            kernel.prepared_async_call(*params, shared_size=layer.lut_size)

        if self.bench or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            print("%7.3f msecs (%s) grid:%s" % (msecs, layer, layer.grid))

    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, repeat=1, size=None):

        assert A.dtype.type == B.dtype.type == C.dtype.type

        flags = 0
        if C.rounding: flags |= 1 | (C.rounding << 16)
        if relu:       flags |= 2

        dima, dimb, dimc = 0,0,0
        ldaz, ldbz, ldcz = 0,0,0
        batch_grid, batch_loops = 1,1

        if len(A.shape) == 3:
            dima = 1
            ldaz = A.strides[0]

        if len(B.shape) == 3:
            dimb = 1
            ldbz = B.strides[0]

        assert dima or dimb, "Tensor A or B must have 3 dims to use batched_dot"

        if len(C.shape) == 3:
            dimc = 1
            ldcz = C.strides[0]
            batch_grid  = C.shape[0]
            assert not dima or A.shape[0] == batch_grid
            assert not dimb or B.shape[0] == batch_grid

        elif dima:
            batch_loops = A.shape[0]
            assert not dimb or B.shape[0] == batch_loops

        elif dimb:
            batch_loops = B.shape[0]
            assert not dima or A.shape[0] == batch_loops

        m = A.shape[0 + dima]
        n = B.shape[1 + dimb]
        k = A.shape[1 + dima]

        assert m == C.shape[0 + dimc]
        assert n == C.shape[1 + dimc]
        assert k == B.shape[0 + dimb]

        lda = max(A.strides[dima:])
        ldb = max(B.strides[dimb:])
        ldc = max(C.strides[dimc:])

        if A.is_trans:
            opA = 't'
            lda *= 8 * A.dtype.itemsize # saves a kernel register
        else:
            opA = 'n'

        if B.is_trans:
            opB = 't'
        else:
            opB = 'n'
            ldb *= 8 * B.dtype.itemsize # saves a kernel register

        op = opA + opB
        assert op != "tt"

        short = min(m,n)
        if batch_loops > 1:
            size = 128
        elif size is None:
            if short % 128 == 0:
                size = 128
            elif short > 32 and short == n: #temp
                size = 64
            else:
                size = 32

        if m >= n:
            if op == "nt":
                size = 128
            sizeA, sizeB = (128,size)
        else:
            if op == "tn":
                size = 128
            # temp till I can write these kernels (coming soon)
            elif size == 64:
                size = 32
            sizeA, sizeB = (size,128)

        gridA   = m // sizeA + (m % sizeA != 0)
        gridB   = n // sizeB + (n % sizeB != 0)
        threads = 256 if size == 128 else 128
        size    = "%dx%d" % (sizeA,sizeB)

        k_vec = 4 if sizeA == 32 or sizeB == 32 else 16

        if  op == "tn" and m % 4 == 0 and n % 4 == 0 or \
            op == "nn" and k % k_vec == 0 and n % 4 == 0 or \
            op == "nt" and k % k_vec == 0:
            op += "_vec"

        # nt and nn are more efficient with k%16==0
        if C.dtype.type is np.float16:
            clss = "hgemm"
        elif C.dtype.type is np.float32:
            clss = "sgemm"
        else:
            raise TypeError("Only floating point dot currently supported.")

        kernel = _get_gemm_kernel(self.cubin_path, clss, op, size)
        params = [
            (batch_grid,gridA,gridB), (threads,1,1), self.stream, _get_rand_state(),
            A.gpudata, B.gpudata, C.gpudata,
            lda, ldb, ldc, m, n, k,
            alpha, beta, flags,
            ldaz, ldbz, ldcz, batch_loops]

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            gflops = (batch_loops * batch_grid * m * n * k * 2.0) / (msecs * 1000000.0)
            print("%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d,%d) loops:%d" %
                  (msecs,gflops,clss,op,m,n,k, size,batch_grid,gridA,gridB,batch_loops))
            if repeat > 1:
                return gflops

        return C

    def dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, repeat=1, size=None):
        """
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C

        relu: if true applied before output (and prior to beta addition)

        size: one of 32x128, 128x32, 64x128, 128x64, 128x128.  Sometimes the fastest tiling isn't chosen for you.
        """
        assert A.dtype.type == B.dtype.type == C.dtype.type

        # one dimention must be contiguous
        assert min(A.strides) == 1
        assert min(B.strides) == 1
        assert min(C.strides) == 1

        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)

        if A.is_trans:
            opA = 't'
            lda *= 8 * A.dtype.itemsize # saves a kernel register
        else:
            opA = 'n'

        if B.is_trans:
            opB = 't'
        else:
            opB = 'n'
            ldb *= 8 * B.dtype.itemsize # saves a kernel register

        op  = opA + opB
        assert op != "tt"

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        assert m == C.shape[0]
        assert n == C.shape[1]
        assert k == B.shape[0]

        # Some basic tile size selection.
        # Your best bet is to benchmark your code with all 3 sizes
        # and manually fine tune the selection for each layer.
        # TODO: Perhaps I'll add an autotuning mode.
        if size is None:
            # find the shorter side
            short = min(m,n)
            # anything bigger than this just use 128
            if short < 384-16:
                # compute remainder of 128
                short128 = short % 128
                # if remainder is more than 112 just use 128
                if 0 < short128 < 112:
                    # to figure out when to use 64 over 32 we need to calc occupancy at 64
                    if 48 < short128 <= 64:
                        occupancy64  = short // 64
                        wide         = max(m,n)
                        occupancy64 *= (wide // 128 + (wide % 128 != 0)) // _get_sm_count()
                        # 64 is only faster than 32 when occupancy is more than 1 warp per scheduler.
                        if occupancy64 > 1:
                            size = 64
                        else:
                            size = 32
                    else:
                        size = 32
                else:
                    size = 128
            # There's a large regime where 64 is faster, but it's hard to characterize
            else:
                size = 128

            # match the kernel to the optimal short size but avoid not implemented kernels
            if m >= n:
                if op == "nt":
                    size = 128
                sizeA, sizeB = (128,size)
            else:
                if op == "tn":
                    size = 128
                # temp till I can write these kernels (coming soon)
                elif size == 64:
                    size = 32
                sizeA, sizeB = (size,128)

            size = "%dx%d" % (sizeA,sizeB)

        else:
            sizeA, sizeB = (int(s) for s in size.split('x'))

        gridA   = m // sizeA + (m % sizeA != 0)
        gridB   = n // sizeB + (n % sizeB != 0)
        threads = 256 if size == "128x128" else 128

        k_vec = 4 if sizeA == 32 or sizeB == 32 else 16

        if  op == "tn" and m % 4 == 0 and n % 4 == 0 or \
            op == "nn" and k % k_vec == 0 and n % 4 == 0 or \
            op == "nt" and k % k_vec == 0:
            op += "_vec"

        # nt and nn are more efficient with k%16==0
        if   C.dtype.type is np.float16:
            clss = "hgemm"
        elif C.dtype.type is np.float32:
            clss = "sgemm"
        else:
            raise TypeError("Only floating point dot currently supported.")

        flags = 0
        if C.rounding: flags |= 1 | (C.rounding << 16)
        if relu:       flags |= 2

        kernel = _get_gemm_kernel(self.cubin_path, clss, op, size)
        params = [
            (1,gridA,gridB), (threads,1,1), self.stream, _get_rand_state(),
            A.gpudata, B.gpudata, C.gpudata,
            lda, ldb, ldc, m, n, k,
            alpha, beta, flags, 0, 0, 0, 0 ]


        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(*params)

        if self.bench or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
            print("%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d)" %
                  (msecs,gflops,clss,op,m,n,k, size,gridA,gridB))
            if repeat > 1:
                return gflops

        return C

    def compensated_sum(self, sum_tensor, cmp_tensor, add_tensor, cmp_scale=1.0, add_scale=1.0):

        if cmp_tensor.kahan_reset and cmp_tensor.kahan_count > cmp_tensor.kahan_reset:
            cmp_scale = 0
            cmp_tensor.kahan_count = 0

        assert sum_tensor.dtype.type == cmp_tensor.dtype.type == add_tensor.dtype.type

        cmp_tensor.kahan_count += 1

        shape, strides = _get_fast_ew_dims(sum_tensor.size)

        kernel   = _get_compensated_sum_kernel(sum_tensor.dtype.str[1:], sum_tensor.rounding > 0)

        kernel.prepared_async_call(
            (shape[0],1,1), (32,1,1), self.stream, _get_rand_state(),
            sum_tensor.gpudata, cmp_tensor.gpudata, add_tensor.gpudata,
            cmp_scale, add_scale,
            strides[0], strides[1],
            shape[1], sum_tensor.rounding)

    def add         (self, a, b, out=None): return OpTreeNode.build("add", a, b, out=out)
    def subtract    (self, a, b, out=None): return OpTreeNode.build("sub", a, b, out=out)
    def multiply    (self, a, b, out=None): return OpTreeNode.build("mul", a, b, out=out)
    def divide      (self, a, b, out=None): return OpTreeNode.build("div", a, b, out=out)
    def true_divide (self, a, b, out=None): return OpTreeNode.build("div", a, b, out=out)
    def power       (self, a, b, out=None): return OpTreeNode.build("pow", a, b, out=out)
    def reciprocal  (self, a,    out=None): return OpTreeNode.build("div", 1, a, out=out)

    def negative    (self, a, out=None): return OpTreeNode.build("neg",  a, None, out=out)
    def sgn         (self, a, out=None): return OpTreeNode.build("sgn",  a, None, out=out)
    def absolute    (self, a, out=None): return OpTreeNode.build("abs",  a, None, out=out)
    def fabs        (self, a, out=None): return OpTreeNode.build("abs",  a, None, out=out)

    def sqrt        (self, a, out=None): return OpTreeNode.build("sqrt", a, None, out=out)
    def square      (self, a, out=None): return OpTreeNode.build("sqr",  a, None, out=out)
    def exp         (self, a, out=None): return OpTreeNode.build("exp",  a, None, out=out)
    def exp2        (self, a, out=None): return OpTreeNode.build("exp2", a, None, out=out)
    def log         (self, a, out=None): return OpTreeNode.build("log",  a, None, out=out)
    def log2        (self, a, out=None): return OpTreeNode.build("log2", a, None, out=out)
    def sig         (self, a, out=None): return OpTreeNode.build("sig",  a, None, out=out)
    def sig2        (self, a, out=None): return OpTreeNode.build("sig2", a, None, out=out)
    def tanh        (self, a, out=None): return OpTreeNode.build("tanh", a, None, out=out)
    def tanh2       (self, a, out=None): return OpTreeNode.build("tanh2",a, None, out=out)

    def finite      (self, a, out=None): return OpTreeNode.build("finite", a, None, out=out)

    def equal         (self, a, b, out=None): return OpTreeNode.build("eq", a, b, out=out)
    def not_equal     (self, a, b, out=None): return OpTreeNode.build("ne", a, b, out=out)
    def less          (self, a, b, out=None): return OpTreeNode.build("lt", a, b, out=out)
    def less_equal    (self, a, b, out=None): return OpTreeNode.build("le", a, b, out=out)
    def greater       (self, a, b, out=None): return OpTreeNode.build("gt", a, b, out=out)
    def greater_equal (self, a, b, out=None): return OpTreeNode.build("ge", a, b, out=out)

    def maximum(self, a, b, out=None): return OpTreeNode.build("maximum", a, b, out=out)
    def minimum(self, a, b, out=None): return OpTreeNode.build("minimum", a, b, out=out)

    def clip(self, a, a_min, a_max, out=None):
        return self.minimum(self.maximum(a, a_min), a_max, out=out)

    def sum(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.sum(self.sum(a, axis=1, out=partial), axis=0, out=out)
        return OpTreeNode.build("sum", a, None, axis=axis, out=out)

    def max(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.max(self.max(a, axis=1, out=partial), axis=0, out=out)
        return OpTreeNode.build("max", a, None, axis=axis, out=out)

    def min(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.min(self.min(a, axis=1, out=partial), axis=0, out=out)
        return OpTreeNode.build("min", a, None, axis=axis, out=out)

    def argmax(self, a, axis=1, out=None, keepdims=True):
        return OpTreeNode.build("argmax", a, None, axis=axis, out=out)

    def argmin(self, a, axis=1, out=None, keepdims=True):
        return OpTreeNode.build("argmin", a, None, axis=axis, out=out)

    def mean(self, a, axis=None, partial=None, out=None, keepdims=True):
        shape = OpTreeNode.shape(a)
        if axis is None:
            assert partial is not None
            return self.multiply(
                        self.sum(self.sum(a, axis=1, out=partial), axis=0),
                        1.0/(shape[0]*shape[1]),
                        out=out)
        return self.multiply(self.sum(a, axis=axis), 1.0/shape[axis], out=out)

    def var(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.mean(
                    self.square(a - self.mean(a, axis=axis, partial=partial, out=partial[0:1,0:1])),
                    axis=axis, partial=partial, out=out)

        return self.mean(self.square(a - self.mean(a, axis=axis)), axis=axis, out=out)

    def std(self, a, axis=None, partial=None, out=None, keepdims=True):
        return self.sqrt(self.var(a, axis=axis, partial=partial, out=out))

    def rand(self, out=None): return OpTreeNode.build("rand", None, None, out=out)

    def dropout(self, keep=0.5, out=None):
        return self.less_equal(self.rand(), keep, out=out)

    def take(self, a, indices, axis, out=None):
        return a.take(indices, axis, out)

    def onehot(self, indices, axis, out=None):
        if axis not in (0,1):
            raise ValueError("bad axis for onehot")
        return OpTreeNode.build("onehot", None, None, idx=indices, axis=axis, out=out)

# For constructing an op tree used in lazy evaluation
class OpTreeNode(tuple):

    def __new__(cls, *args):

        return tuple.__new__(cls, args)

    @staticmethod
    def build(op, a, b, out=None, **kwargs):

        for arg in (a,b):
            if not isinstance(arg, (int, float, GPUTensor, OpTreeNode, type(None))):
                return NotImplemented

        op_dict = { "op" : op }
        op_dict.update(kwargs)

        node = OpTreeNode(op_dict, a, b)

        # execute explicit assignment
        if op == "assign":
            return node.execute()

        # passing in an out value counts as assignment
        if out is not None:
            return OpTreeNode({ "op" : "assign" }, out, node).execute()

        # delay execution until assignment
        return node

    def execute(self):

        stack = self.traverse(list())

        return call_compound_kernel(_get_rand_state(), *stack)

    # post order walk op tree and produce postfix stack
    def traverse(self, stack):

        # Left
        if type(self[1]) is OpTreeNode:
            self[1].traverse(stack)
        elif self[1] is not None:
            stack.append(self[1])

        # Right
        if type(self[2]) is OpTreeNode:
            self[2].traverse(stack)
        elif self[2] is not None:
            stack.append(self[2])

        stack.append(self[0])

        return stack

    @staticmethod
    def shape(node):

        if type(node) is GPUTensor:
            return node.shape

        if type(node) is OpTreeNode:

            max_shape = [1,1]
            stack = node.traverse(list())
            for item in stack:
                if type(item) is GPUTensor:
                    for i in range(2):
                        max_shape[i] = max(max_shape[i], item.shape[i])
            return tuple(max_shape)

        #scalar
        return (1,1)

    def __add__      (self, other): return self.build("add", self, other)
    def __sub__      (self, other): return self.build("sub", self, other)
    def __mul__      (self, other): return self.build("mul", self, other)
    def __div__      (self, other): return self.build("div", self, other)
    def __truediv__  (self, other): return self.build("div", self, other)
    def __pow__      (self, other): return self.build("pow", self, other)
    def __radd__     (self, other): return self.build("add", other, self)
    def __rsub__     (self, other): return self.build("sub", other, self)
    def __rmul__     (self, other): return self.build("mul", other, self)
    def __rdiv__     (self, other): return self.build("div", other, self)
    def __rtruediv__ (self, other): return self.build("div", other, self)
    def __rpow__     (self, other): return self.build("pow", other, self)
    def __eq__       (self, other): return self.build("eq",  self, other)
    def __ne__       (self, other): return self.build("ne",  self, other)
    def __lt__       (self, other): return self.build("lt",  self, other)
    def __le__       (self, other): return self.build("le",  self, other)
    def __gt__       (self, other): return self.build("gt",  self, other)
    def __ge__       (self, other): return self.build("ge",  self, other)
    def __abs__      (self):        return self.build("abs", self,  None)
    def __neg__      (self):        return self.build("neg", self,  None)

    #def __nonzero__  (self): raise ValueError("The truth value of an array with more than one element is ambiguous.")

# Note the strides computed here do not include the dtype.itemsize
def _contiguous_strides(shape):
    if shape:
        strides = [1]
        for s in shape[:0:-1]:
            strides.append(strides[-1]*s)
        return tuple(strides[::-1])
    else:
        return ()

@context_dependent_memoize
def _get_scratch_data(scratch_size):
    return drv.mem_alloc(scratch_size*4)

@context_dependent_memoize
def _get_rand_state():
    # initialize our common pool of randomness (1/4 MB):
    # MAX_THREADS_PER_MULTIPROCESSOR * 32 SMs (32 to be somewhat future proof
    # and power of two). This size is currently hardcoded in the kernels,
    # to be parameterized ...
    rand_init  = np.random.random_integers(0,2**32-1,(2048*32,)).astype(np.uint32)
    rand_state = drv.mem_alloc(rand_init.nbytes)
    drv.memcpy_htod(rand_state, rand_init)
    return rand_state

@context_dependent_memoize
def _get_events():
    return (drv.Event(), drv.Event())

@context_dependent_memoize
def _get_module(path, clss, op, size=None):

    size = "" if size is None else "_" + size
    cubin = "{0}_{1}{2}.cubin".format(clss, op, size)
    return drv.module_from_file(os.path.join(path, cubin))

@context_dependent_memoize
def _get_gemm_kernel(path, clss, op, size):
    module = _get_module(path, clss, op, size)
    kernel = "{0}_{1}_{2}".format(clss, op, size)
    func   = module.get_function(kernel)
    func.prepare("PPPPIIIIIIffIIIII")
    #print("Loaded: ", kernel)
    return func

_conv_sig = {
    "fprop_K64_N64"   : "PPPfIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    "bprop_C64_N64"   : "PPPfIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    "bprop_C32_N64"   : "PPPfIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    "updat_C128_K64"  : "PPPfIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    "updat_C128_K128" : "PPPfIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
}

@context_dependent_memoize
def _get_conv_kernel(path, clss, op, size):
    module = _get_module(path, clss, op, size)
    kernel = "{0}_{1}_{2}".format(clss, op, size)
    func   = module.get_function(kernel)
    func.prepare(_conv_sig["{0}_{1}".format(op, size)])
    #print("Loaded: ", kernel)
    return func

@context_dependent_memoize
def _get_pool_kernel(path, clss, op):

    module = _get_module(path, clss, op)
    kernel = "{0}_{1}".format(clss, op)
    func   = module.get_function(kernel)
    func.prepare("PPPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIf")
    #print("Loaded: ", kernel)
    return func

# debugging tool
# import re
# import traceback as tb

# nrv_re = re.compile(r'nervanagpu\.py$')
# def print_trace():
#     caller = None
#     for frame in tb.extract_stack():
#         if GPUTensor.nrv_re.search(frame[0]):
#             break
#         caller = (frame[0],frame[1])
#     print caller

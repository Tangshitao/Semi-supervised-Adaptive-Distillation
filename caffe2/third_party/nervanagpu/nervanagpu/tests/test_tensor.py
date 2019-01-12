
import nervanagpu as ng
import numpy as np
import pycuda.autoinit
from nose.plugins.attrib import attr
from nervanagpu.util.testing import assert_tensor_equal, assert_tensor_near_equal

class TestTensor(object):

	def setup(self):

		self.gpu = ng.NervanaGPU(stochastic_round=False)
		self.dims = (1024,1024)

	def init_helper(self, lib, inA, inB, dtype):

		A = lib.array(inA, dtype=dtype)
		B = lib.array(inB, dtype=dtype)
		C = lib.empty(inB.shape, dtype=dtype)

		return A, B, C

	def math_helper(self, lib, op, inA, inB, dtype):

		A, B, C = self.init_helper(lib, inA, inB, dtype)

		if op == '+':
			C[:] = A + B
		elif op == '-':
			C[:] = A - B
		elif op == '*':
			C[:] = A * B
		elif op == '/':
			C[:] = A / B
		elif op == '>':
			C[:] = A > B
		elif op == '>=':
			C[:] = A >= B
		elif op == '<':
			C[:] = A < B
		elif op == '<=':
			C[:] = A <= B

		return C

	def compare_helper(self, op, inA, inB, dtype):

		numpy_result = self.math_helper(np, op, inA, inB, dtype=np.float32)

		if np.dtype(dtype).kind == 'i' or np.dtype(dtype).kind == 'u':
			numpy_result = np.around(numpy_result)
			numpy_result = numpy_result.clip(np.iinfo(dtype).min, np.iinfo(dtype).max)
		numpy_result = numpy_result.astype(dtype)

		nervanaGPU_result = self.math_helper(self.gpu, op, inA, inB, dtype=dtype)

		assert_tensor_near_equal(numpy_result, nervanaGPU_result, 1e-5)


	def rand_unif(self, dtype, dims):
		if np.dtype(dtype).kind == 'f':
			return np.random.uniform(-1, 1, dims).astype(dtype)
		else:
			iinfo = np.iinfo(dtype)
			return np.around(np.random.uniform(iinfo.min, iinfo.max, dims)) \
					.clip(iinfo.min, iinfo.max)

	def test_math(self):

		for dtype in (np.float32, np.float16, np.int8, np.uint8):

			randA = self.rand_unif(dtype, self.dims)
			randB = self.rand_unif(dtype, self.dims)

			self.compare_helper('+', randA, randB, dtype)
			self.compare_helper('-', randA, randB, dtype)
			self.compare_helper('*', randA, randB, dtype)
			#self.compare_helper('/', randA, randB, dtype)
			self.compare_helper('>', randA, randB, dtype)
			self.compare_helper('>=', randA, randB, dtype)
			self.compare_helper('<', randA, randB, dtype)
			self.compare_helper('<=', randA, randB, dtype)

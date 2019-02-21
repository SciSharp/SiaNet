/*
Copyright (c) 2015, ArrayFire
Copyright (c) 2015, Steven Burns (royalstream@hotmail.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of arrayfire_dotnet nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

using SiaNet.Backend.ArrayFire.Interop;

namespace SiaNet.Backend.ArrayFire
{

	public static class Matrix
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Multiply(NDArray lhs, NDArray rhs, MatMulOp lop = MatMulOp.None, MatMulOp rop = MatMulOp.None)
		{
			IntPtr ptr;
			Internal.VERIFY(AFBlas.af_matmul(out ptr, lhs._ptr, rhs._ptr, (af_mat_prop)lop, (af_mat_prop)rop));
			return new NDArray(ptr);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void TransposeInPlace(NDArray arr, bool conjugate)
		{
			Internal.VERIFY(AFBlas.af_transpose_inplace(arr._ptr, conjugate));
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Transpose(NDArray arr, bool conjugate)
		{
			IntPtr ptr;
			Internal.VERIFY(AFBlas.af_transpose(out ptr, arr._ptr, conjugate));
			return new NDArray(ptr);
		}

		/*
		public static double Det(Array<double> arr)
		{
			double r, i;
			Internal.VERIFY(AFLapack.af_det(out r, out i, arr._ptr));
			return r;
		}

		public static float Det(Array<float> arr)
		{
			double r, i;
			Internal.VERIFY(AFLapack.af_det(out r, out i, arr._ptr));
			return (float)r;
		}*/

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Complex Det(NDArray arr)
		{
			double r, i;
			Internal.VERIFY(AFLapack.af_det(out r, out i, arr._ptr));
			return new Complex(r, i);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Inverse(NDArray arr)
		{
			IntPtr ptr;
			Internal.VERIFY(AFLapack.af_inverse(out ptr, arr._ptr, af_mat_prop.AF_MAT_NONE));
			return new NDArray(ptr);
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray ExtractDiagonal(NDArray arr, int diagonalIndex = 0)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_diag_extract(out ptr, arr._ptr, diagonalIndex));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateDiagonal(NDArray arr, int diagonalIndex = 0)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_diag_create(out ptr, arr._ptr, diagonalIndex));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Lower(NDArray arr, bool unitDiagonal = false)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_lower(out ptr, arr._ptr, unitDiagonal));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Upper(NDArray arr, bool unitDiagonal = false)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_upper(out ptr, arr._ptr, unitDiagonal));
            return new NDArray(ptr);
        }

        #region Convenience methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Identity<T>(int dim0, int dim1) { return Data.Identity<T>(dim0, dim1); }

        public static T[,] GetData<T>(NDArray arr) { return Data.GetData2D<T>(arr); }
        #endregion
    }
}

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
	// we can't make Arith static because Array inherits from it (so the F# Core.Operators free functions work correctly)
	public /*static*/ class Arith
	{
		protected Arith() { }

        #region Mathematical Functions
#if _
	for (\w+) in
		Sin Sinh Asin Asinh
		Cos Cosh Acos Acosh
		Tan Tanh Atan Atanh
		Exp Expm1 Log Log10 Log1p Log2 Erf Erfc
		Sqrt Pow2 Cbrt
		LGamma TGamma
		Abs Sigmoid Factorial
		Round Trunc Floor Ceil
	do
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Array $1(Array arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_$L1(out ptr, arr._ptr)); return new Array(ptr); }
#endif
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Arg(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_arg(out ptr, arr._ptr)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Sin(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_sin(out ptr, arr._ptr)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Sign(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_sign(out ptr, arr._ptr)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Sinh(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_sinh(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Asin(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_asin(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Asinh(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_asinh(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Cos(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_cos(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Cosh(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_cosh(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Acos(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_acos(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Acosh(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_acosh(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Tan(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_tan(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Tanh(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_tanh(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Atan(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_atan(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Atanh(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_atanh(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Exp(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_exp(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Expm1(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_expm1(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Log(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_log(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Log10(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_log10(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Log1p(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_log1p(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Log2(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_log2(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Erf(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_erf(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Erfc(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_erfc(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Sqrt(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_sqrt(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Pow2(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_pow2(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Cbrt(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_cbrt(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray LGamma(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_lgamma(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray TGamma(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_tgamma(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Abs(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_abs(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Sigmoid(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_sigmoid(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Factorial(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_factorial(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Round(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_round(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Trunc(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_trunc(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Floor(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_floor(out ptr, arr._ptr)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Ceil(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_ceil(out ptr, arr._ptr)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Clamp(NDArray arr, NDArray lo, NDArray hi) { IntPtr ptr; Internal.VERIFY(AFArith.af_clamp(out ptr, arr._ptr, lo._ptr, hi._ptr, false)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Neg(NDArray arr) { IntPtr ptr; Internal.VERIFY(AFArith.af_neg(out ptr, arr._ptr)); return new NDArray(ptr); }
#if _
	for (\w+) in
		Atan2 Rem Pow
	do
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Array $1(Array lhs, Array rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_$L1(out ptr, lhs._ptr, rhs._ptr, false)); return new Array(ptr); }
#endif
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Atan2(NDArray lhs, NDArray rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_atan2(out ptr, lhs._ptr, rhs._ptr, false)); return new NDArray(ptr); }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Rem(NDArray lhs, NDArray rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_rem(out ptr, lhs._ptr, rhs._ptr, false)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray EqualTo(NDArray lhs, NDArray rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_eq(out ptr, lhs._ptr, rhs._ptr, false)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Pow(NDArray lhs, NDArray rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_pow(out ptr, lhs._ptr, rhs._ptr, false)); return new NDArray(ptr); }

        public static NDArray MaxOf(NDArray lhs, NDArray rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_maxof(out ptr, lhs._ptr, rhs._ptr, false)); return new NDArray(ptr); }

        public static NDArray MinOf(NDArray lhs, NDArray rhs) { IntPtr ptr; Internal.VERIFY(AFArith.af_minof(out ptr, lhs._ptr, rhs._ptr, false)); return new NDArray(ptr); }
        #endregion
    }
}

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
    public static class Algorithm
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Sum(NDArray arr)
        {
            double r, i;
            Internal.VERIFY(AFAlgorithm.af_sum_all(out r, out i, arr._ptr));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Sum(NDArray arr, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFAlgorithm.af_sum(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Prod(NDArray arr)
        {
            double r, i;
            Internal.VERIFY(AFAlgorithm.af_product_all(out r, out i, arr._ptr));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Prod(NDArray arr, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFAlgorithm.af_product(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Max(NDArray arr)
        {
            double r, i;
            Internal.VERIFY(AFAlgorithm.af_max_all(out r, out i, arr._ptr));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Max(NDArray arr, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFAlgorithm.af_max(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Min(NDArray arr)
        {
            double r, i;
            Internal.VERIFY(AFAlgorithm.af_min_all(out r, out i, arr._ptr));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Min(NDArray arr, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFAlgorithm.af_min(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Mean(NDArray arr)
        {
            double r, i;
            Internal.VERIFY(AFStatistics.af_mean_all(out r, out i, arr._ptr));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Mean(NDArray arr, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFStatistics.af_mean(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex StdDev(NDArray arr)
        {
            double r, i;
            Internal.VERIFY(AFStatistics.af_stdev_all(out r, out i, arr._ptr));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray StdDev(NDArray arr, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFStatistics.af_stdev(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Var(NDArray arr, bool isbiased = false)
        {
            double r, i;
            Internal.VERIFY(AFStatistics.af_var_all(out r, out i, arr._ptr, isbiased));
            return new Complex(r, i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Var(NDArray arr, int dim, bool isbiased = false)
        {
            IntPtr ptr;
            Internal.VERIFY(AFStatistics.af_var(out ptr, arr._ptr, isbiased, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray TopK(NDArray arr, int k, int dim, int order)
        {
            IntPtr ptr;
            IntPtr idx;
            Internal.VERIFY(AFStatistics.af_topk(out ptr, out idx, arr._ptr, k, dim, order));
            return new NDArray(idx);
        }
    }
}

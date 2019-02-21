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
	public static class Data
	{

		#region Create array from host data
#if _
    for (\w+)=(\w+) in
        b8=bool c64=Complex f32=float f64=double s32=int s64=long u32=uint u64=ulong u8=byte s16=short u16=ushort
    do
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array CreateArray($2[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.$1)); return new Array(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array CreateArray($2[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.$1)); return new Array(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array CreateArray($2[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.$1)); return new Array(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array CreateArray($2[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.$1)); return new Array(ptr); }
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(bool[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.b8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(bool[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.b8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(bool[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.b8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(bool[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.b8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(Complex[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.c64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(Complex[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.c64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(Complex[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.c64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(Complex[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.c64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(float[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.f32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(float[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.f32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(float[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.f32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(float[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.f32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(double[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.f64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(double[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.f64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(double[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.f64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(double[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.f64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(int[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.s32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(int[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.s32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(int[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.s32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(int[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.s32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(long[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.s64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(long[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.s64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(long[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.s64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(long[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.s64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(uint[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.u32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(uint[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.u32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(uint[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.u32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(uint[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.u32)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ulong[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.u64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ulong[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.u64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ulong[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.u64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ulong[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.u64)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(byte[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.u8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(byte[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.u8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(byte[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.u8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(byte[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.u8)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(short[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.s16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(short[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.s16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(short[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.s16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(short[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.s16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ushort[] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.Length }, af_dtype.u16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ushort[,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1) }, af_dtype.u16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ushort[,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) }, af_dtype.u16)); return new NDArray(ptr); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray CreateArray(ushort[,,,] data) { IntPtr ptr; Internal.VERIFY(AFArray.af_create_array(out ptr, data, (uint)data.Rank, new long[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) }, af_dtype.u16)); return new NDArray(ptr); }
#endif
        #endregion

        #region Write array from host data
#if _
    for (\w+)=(\w+) in
        b8=bool c64=Complex f32=float f64=double s32=int s64=long u32=uint u64=ulong u8=byte s16=short u16=ushort
    do
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(Array arr, $2[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(Array arr, $2[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(Array arr, $2[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(Array arr, $2[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, bool[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, bool[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, bool[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, bool[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, Complex[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, Complex[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, Complex[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, Complex[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, float[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, float[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, float[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, float[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, double[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, double[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, double[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, double[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, int[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, int[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, int[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, int[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, long[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, long[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, long[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, long[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, uint[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, uint[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, uint[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, uint[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ulong[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ulong[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ulong[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ulong[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, byte[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, byte[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, byte[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, byte[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, short[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, short[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, short[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, short[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ushort[] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ushort[,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ushort[,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void WriteArray(NDArray arr, ushort[,,,] data) { Internal.VERIFY(AFArray.af_write_array(arr._ptr, data, Internal.sizeOfArray(data), af_source.afHost)); }
#endif
		#endregion

		#region Random Arrays
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray RandUniform<T>(params int[] dims)
		{
			IntPtr ptr;
			Internal.VERIFY(AFData.af_randu(out ptr, (uint)dims.Length, Internal.toLongArray(dims), Internal.toDType<T>()));
			return new NDArray(ptr);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray RandNormal<T>(params int[] dims)
		{
			IntPtr ptr;
			Internal.VERIFY(AFData.af_randn(out ptr, (uint)dims.Length, Internal.toLongArray(dims), Internal.toDType<T>()));
			return new NDArray(ptr);
		}

		public static ulong RandSeed
		{
			get { ulong value; Internal.VERIFY(AFData.af_get_seed(out value)); return value; }
			set { Internal.VERIFY(AFData.af_set_seed(value)); }
		}
		#endregion

		#region Constant, Iota, Range, Identity
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Constant<T>(T value, params int[] dims)
		{
			IntPtr ptr;
			object boxval = value;
			af_dtype dtype = Internal.toDType<T>();
			switch (dtype)
			{
				case af_dtype.u64:
					Internal.VERIFY(AFData.af_constant_ulong(out ptr, (ulong)boxval, (uint)dims.Length, Internal.toLongArray(dims)));
					break;
				case af_dtype.s64:
					Internal.VERIFY(AFData.af_constant_long(out ptr, (long)boxval, (uint)dims.Length, Internal.toLongArray(dims)));
					break;
				case af_dtype.c64:
					Complex z = (Complex)boxval;
					Internal.VERIFY(AFData.af_constant_complex(out ptr, z.Real, z.Imaginary, (uint)dims.Length, Internal.toLongArray(dims), dtype));
					break;
				default:
					Internal.VERIFY(AFData.af_constant(out ptr, (double)Convert.ChangeType(boxval, typeof(double)), (uint)dims.Length, Internal.toLongArray(dims), dtype));
					break;
			}
			return new NDArray(ptr);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Iota<T>(int[] dims, int[] tiles)
		{
			IntPtr ptr;
			Internal.VERIFY(AFData.af_iota(out ptr, (uint)dims.Length, Internal.toLongArray(dims), (uint)tiles.Length, Internal.toLongArray(tiles), Internal.toDType<T>()));
			return new NDArray(ptr);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Iota<T>(params int[] dims)
		{
			return Iota<T>(dims, new int[] { 1 });
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray RangeAlong<T>(int seq_dim, params int[] dims)
		{
			IntPtr ptr;
			Internal.VERIFY(AFData.af_range(out ptr, (uint)dims.Length, Internal.toLongArray(dims), seq_dim, Internal.toDType<T>()));
			return new NDArray(ptr);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Range<T>(params int[] dims)
		{
			return RangeAlong<T>(-1, dims); // -1 is the default according to af_range's documentation
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Identity<T>(params int[] dims)
		{
			IntPtr ptr;
			Internal.VERIFY(AFData.af_identity(out ptr, (uint)dims.Length, Internal.toLongArray(dims), Internal.toDType<T>()));
			return new NDArray(ptr);
		}
		#endregion

		#region Complex Arrays from real data
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray CreateComplexArray(NDArray real, NDArray imag = null)
		{
			IntPtr ptr;
			if (imag != null) Internal.VERIFY(AFArith.af_cplx2(out ptr, real._ptr, imag._ptr, false));
			else Internal.VERIFY(AFArith.af_cplx(out ptr, real._ptr));
			return new NDArray(ptr);
		}
		#endregion

		#region Get the array inner data
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static T[] GetData<T>(NDArray arr) // not called GetData1D because it works for arrays of any dimensionality
		{
			T[] data = new T[arr.ElemCount];
			Internal.VERIFY(Internal.getData<T>(data, arr._ptr));
			return data;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static T[,] GetData2D<T>(NDArray arr) // only works for 2D arrays
		{
			int[] dims = arr.Dimensions;
			if (dims.Length > 2) throw new NotSupportedException("This array has more than two dimensions");
			T[,] data;
			if (dims.Length == 1) // column vector
				data = new T[dims[0], 1];
			else
				data = new T[dims[0], dims[1]];
			Internal.VERIFY(Internal.getData<T>(data, arr._ptr));
			return data;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static T[,,] GetData3D<T>(NDArray arr) // only works for 3D arrays
		{
			int[] dims = arr.Dimensions;
			if (dims.Length > 3) throw new NotSupportedException("This array has more than three dimensions");
			T[,,] data;
			if (dims.Length == 1)
				data = new T[dims[0], 1, 1];
			else if (dims.Length == 2)
				data = new T[dims[0], dims[1], 1];
			else
				data = new T[dims[0], dims[1], dims[2]];
			Internal.VERIFY(Internal.getData<T>(data, arr._ptr));
			return data;
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static T[,,,] GetData4D<T>(NDArray arr)
		{
			int[] dims = arr.Dimensions;
			if (dims.Length > 4) throw new NotSupportedException("This array has more than four dimensions");
			T[,,,] data;
			if (dims.Length == 1)
				data = new T[dims[0], 1, 1, 1];
			else if (dims.Length == 2)
				data = new T[dims[0], dims[1], 1, 1];
			else if (dims.Length == 3)
				data = new T[dims[0], dims[1], dims[2], 1];
			else
				data = new T[dims[0], dims[1], dims[2], dims[3]];
			Internal.VERIFY(Internal.getData<T>(data, arr._ptr));
			return data;
		}
		#endregion

		#region Casting
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static NDArray Cast<X>(NDArray arr)
		{
			IntPtr ptr;
			Internal.VERIFY(AFArith.af_cast(out ptr, arr._ptr, Internal.toDType<X>()));
			return new NDArray(ptr);
		}
        #endregion

        #region Copying
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Copy(NDArray arr)
        {
            IntPtr ptr;
            Internal.VERIFY(AFArray.af_copy_array(out ptr, arr._ptr));
            return new NDArray(ptr);
        }
        #endregion

        #region Move Reorder
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Flat(NDArray arr)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_flat(out ptr, arr._ptr));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Flip(NDArray arr, uint dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_flip(out ptr, arr._ptr, dim));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Join(NDArray first, NDArray second, int dim)
        {
            IntPtr ptr;
            Internal.VERIFY(AFData.af_join(out ptr, dim, first._ptr, second._ptr));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Join(NDArray[] list, int dim)
        {
            IntPtr ptr;
            IntPtr[] listPtr = new IntPtr[list.Length];
            for (int i = 0; i < list.Length; i++)
            {
                listPtr[i] = list[i]._ptr;
            }

            Internal.VERIFY(AFData.af_join_many(out ptr, dim, (uint)list.Length, listPtr));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Reorder(NDArray arr, params uint[] dims)
        {
            IntPtr ptr;
            if (dims.Length < 2)
                throw new ArgumentException("Specify at least 2 axis");

            uint z = dims.Length >= 3 ? dims[2] : 1;
            uint w = dims.Length >= 4 ? dims[3] : 1;
            Internal.VERIFY(AFData.af_reorder(out ptr, arr._ptr, dims[0], dims[1], z, w));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray ModDims(NDArray arr, params long[] dims)
        {
            IntPtr ptr;

            Internal.VERIFY(AFData.af_moddims(out ptr, arr._ptr, (uint)dims.Length, dims));
            return new NDArray(ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray Tile(NDArray arr, params uint[] dims)
        {
            IntPtr ptr;
            if (dims.Length < 1)
                throw new ArgumentException("Specify at least 1 axis");

            uint y = dims.Length >= 2 ? dims[1] : 1;
            uint z = dims.Length >= 3 ? dims[2] : 1;
            uint w = dims.Length >= 4 ? dims[3] : 1;

            Internal.VERIFY(AFData.af_tile(out ptr, arr._ptr, dims[0], y, z, w));
            return new NDArray(ptr);
        }
        #endregion
    }
}

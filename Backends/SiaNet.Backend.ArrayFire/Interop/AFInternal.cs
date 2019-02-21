// This file was automatically generated using the AutoGenTool project
// If possible, edit the tool instead of editing this file directly

using System;
using System.Text;
using System.Numerics;
using System.Security;
using System.Runtime.InteropServices;

namespace SiaNet.Backend.ArrayFire.Interop
{
	[SuppressUnmanagedCodeSecurity]
	public static class AFInternal
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] bool[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] Complex[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] float[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] double[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] int[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] long[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] uint[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ulong[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] byte[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] short[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ushort[] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] bool[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] Complex[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] float[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] double[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] int[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] long[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] uint[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ulong[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] byte[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] short[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ushort[,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] bool[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] Complex[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] float[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] double[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] int[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] long[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] uint[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ulong[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] byte[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] short[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ushort[,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] bool[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] Complex[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] float[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] double[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] int[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] long[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] uint[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ulong[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] byte[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] short[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_strided_array(out IntPtr array_arr, [In] ushort[,,,] data, long dim_offset, uint ndims, [In] long[] dim_dims, [In] long[] dim_strides, af_dtype ty, af_source location);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_strides(out long dim_s0, out long dim_s1, out long dim_s2, out long dim_s3, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_offset(out long dim_offset, IntPtr array_arr);

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_raw_ptr(???void **ptr???, IntPtr array_arr); */

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_linear(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_owner(out bool result, IntPtr array_arr);
	}
}

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
	public static class AFArray
	{
        [DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] bool[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] Complex[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] float[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] double[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] int[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] long[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] uint[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ulong[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] byte[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] short[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ushort[] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] bool[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] Complex[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] float[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] double[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] int[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] long[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] uint[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ulong[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] byte[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] short[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ushort[,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] bool[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] Complex[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] float[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] double[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] int[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] long[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] uint[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ulong[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] byte[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] short[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ushort[,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] bool[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] Complex[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] float[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] double[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] int[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] long[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] uint[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ulong[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] byte[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] short[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_array(out IntPtr array_arr, [In] ushort[,,,] data, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_handle(out IntPtr array_arr, uint ndims, [In] long[] dim_dims, af_dtype type);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_copy_array(out IntPtr array_arr, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] bool[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] Complex[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] float[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] double[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] int[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] long[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] uint[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ulong[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] byte[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] short[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ushort[] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] bool[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] Complex[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] float[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] double[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] int[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] long[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] uint[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ulong[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] byte[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] short[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ushort[,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] bool[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] Complex[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] float[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] double[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] int[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] long[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] uint[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ulong[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] byte[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] short[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ushort[,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] bool[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] Complex[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] float[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] double[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] int[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] long[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] uint[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ulong[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] byte[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] short[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_write_array(IntPtr array_arr, [In] ushort[,,,] data, UIntPtr size_bytes, af_source src);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] bool[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] Complex[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] float[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] double[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] int[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] long[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] uint[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ulong[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] byte[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] short[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ushort[] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] bool[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] Complex[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] float[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] double[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] int[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] long[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] uint[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ulong[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] byte[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] short[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ushort[,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] bool[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] Complex[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] float[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] double[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] int[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] long[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] uint[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ulong[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] byte[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] short[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ushort[,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] bool[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] Complex[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] float[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] double[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] int[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] long[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] uint[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ulong[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] byte[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] short[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ptr([Out] ushort[,,,] data, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_release_array(IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_retain_array(out IntPtr array_out, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_data_ref_count(out int use_count, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_eval(IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_elements(out long dim_elems, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_type(out af_dtype type, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_dims(out long dim_d0, out long dim_d1, out long dim_d2, out long dim_d3, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_numdims(out uint result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_empty(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_scalar(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_row(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_column(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_vector(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_complex(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_real(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_double(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_single(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_realfloating(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_floating(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_integer(out bool result, IntPtr array_arr);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_bool(out bool result, IntPtr array_arr);
	}
}

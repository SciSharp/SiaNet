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
	public static class AFStatistics
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_mean(out IntPtr array_out, IntPtr array_in, long dim_dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_mean_weighted(out IntPtr array_out, IntPtr array_in, IntPtr array_weights, long dim_dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_var(out IntPtr array_out, IntPtr array_in, bool isbiased, long dim_dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_var_weighted(out IntPtr array_out, IntPtr array_in, IntPtr array_weights, long dim_dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_stdev(out IntPtr array_out, IntPtr array_in, long dim_dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_cov(out IntPtr array_out, IntPtr array_X, IntPtr array_Y, bool isbiased);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_median(out IntPtr array_out, IntPtr array_in, long dim_dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_mean_all(out double real, out double imag, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_mean_all_weighted(out double real, out double imag, IntPtr array_in, IntPtr array_weights);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_var_all(out double realVal, out double imagVal, IntPtr array_in, bool isbiased);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_var_all_weighted(out double realVal, out double imagVal, IntPtr array_in, IntPtr array_weights);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_stdev_all(out double real, out double imag, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_median_all(out double realVal, out double imagVal, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_corrcoef(out double realVal, out double imagVal, IntPtr array_X, IntPtr array_Y);

        [DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        public static extern af_err af_topk(out IntPtr array_out, out IntPtr array_idx, IntPtr array_in, int k, int dim, int order);
    }
}

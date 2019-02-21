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
	public static class AFSignal
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_approx1(out IntPtr array_out, IntPtr array_in, IntPtr array_pos, af_interp_type method, float offGrid);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_approx2(out IntPtr array_out, IntPtr array_in, IntPtr array_pos0, IntPtr array_pos1, af_interp_type method, float offGrid);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_odim0);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft_inplace(IntPtr array_in, double norm_factor);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft2(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_odim0, long dim_odim1);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft2_inplace(IntPtr array_in, double norm_factor);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft3(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_odim0, long dim_odim1, long dim_odim2);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft3_inplace(IntPtr array_in, double norm_factor);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_ifft(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_odim0);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_ifft_inplace(IntPtr array_in, double norm_factor);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_ifft2(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_odim0, long dim_odim1);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_ifft2_inplace(IntPtr array_in, double norm_factor);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_ifft3(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_odim0, long dim_odim1, long dim_odim2);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_ifft3_inplace(IntPtr array_in, double norm_factor);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft_r2c(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_pad0);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft2_r2c(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_pad0, long dim_pad1);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft3_r2c(out IntPtr array_out, IntPtr array_in, double norm_factor, long dim_pad0, long dim_pad1, long dim_pad2);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft_c2r(out IntPtr array_out, IntPtr array_in, double norm_factor, bool is_odd);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft2_c2r(out IntPtr array_out, IntPtr array_in, double norm_factor, bool is_odd);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft3_c2r(out IntPtr array_out, IntPtr array_in, double norm_factor, bool is_odd);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_convolve1(out IntPtr array_out, IntPtr array_signal, IntPtr array_filter, af_conv_mode mode, af_conv_domain domain);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_convolve2(out IntPtr array_out, IntPtr array_signal, IntPtr array_filter, af_conv_mode mode, af_conv_domain domain);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_convolve3(out IntPtr array_out, IntPtr array_signal, IntPtr array_filter, af_conv_mode mode, af_conv_domain domain);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_convolve2_sep(out IntPtr array_out, IntPtr array_col_filter, IntPtr array_row_filter, IntPtr array_signal, af_conv_mode mode);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft_convolve1(out IntPtr array_out, IntPtr array_signal, IntPtr array_filter, af_conv_mode mode);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft_convolve2(out IntPtr array_out, IntPtr array_signal, IntPtr array_filter, af_conv_mode mode);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fft_convolve3(out IntPtr array_out, IntPtr array_signal, IntPtr array_filter, af_conv_mode mode);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_fir(out IntPtr array_y, IntPtr array_b, IntPtr array_x);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_iir(out IntPtr array_y, IntPtr array_b, IntPtr array_a, IntPtr array_x);
	}
}

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
	public static class AFLapack
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_svd(out IntPtr array_u, out IntPtr array_s, out IntPtr array_vt, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_svd_inplace(out IntPtr array_u, out IntPtr array_s, out IntPtr array_vt, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_lu(out IntPtr array_lower, out IntPtr array_upper, out IntPtr array_pivot, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_lu_inplace(out IntPtr array_pivot, IntPtr array_in, bool is_lapack_piv);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_qr(out IntPtr array_q, out IntPtr array_r, out IntPtr array_tau, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_qr_inplace(out IntPtr array_tau, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_cholesky(out IntPtr array_out, out int info, IntPtr array_in, bool is_upper);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_cholesky_inplace(out int info, IntPtr array_in, bool is_upper);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_solve(out IntPtr array_x, IntPtr array_a, IntPtr array_b, af_mat_prop options);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_solve_lu(out IntPtr array_x, IntPtr array_a, IntPtr array_piv, IntPtr array_b, af_mat_prop options);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_inverse(out IntPtr array_out, IntPtr array_in, af_mat_prop options);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_rank(out uint rank, IntPtr array_in, double tol);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_det(out double det_real, out double det_imag, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_norm(out double output, IntPtr array_in, af_norm_type type, double p, double q);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_is_lapack_available(out bool output);
	}
}

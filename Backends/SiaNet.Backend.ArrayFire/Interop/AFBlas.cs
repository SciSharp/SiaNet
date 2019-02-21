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
	public static class AFBlas
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_matmul(out IntPtr array_out, IntPtr array_lhs, IntPtr array_rhs, af_mat_prop optLhs, af_mat_prop optRhs);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_dot(out IntPtr array_out, IntPtr array_lhs, IntPtr array_rhs, af_mat_prop optLhs, af_mat_prop optRhs);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_transpose(out IntPtr array_out, IntPtr array_in, bool conjugate);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_transpose_inplace(IntPtr array_in, bool conjugate);
	}
}

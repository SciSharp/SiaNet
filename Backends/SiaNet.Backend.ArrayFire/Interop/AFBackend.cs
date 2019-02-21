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
	public static class AFBackend
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_set_backend(af_backend bknd);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_backend_count(out uint num_backends);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_available_backends(out int backends);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_backend_id(out af_backend backend, IntPtr array_in);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_active_backend(out af_backend backend);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_get_device_id(out int device, IntPtr array_in);
	}
}

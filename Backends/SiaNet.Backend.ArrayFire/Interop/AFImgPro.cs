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
	public static class AFImgPro
    {
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_unwrap(out IntPtr array_out, IntPtr array_in, uint wx, uint wy, uint sx, uint sy, uint px, uint py, bool is_column);

        [DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        public static extern af_err af_wrap(out IntPtr array_out, IntPtr array_in, uint ox, uint oy, uint wx, uint wy, uint sx, uint sy, uint px, uint py, bool is_column);

        [DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
        public static extern af_err af_load_image_memory(out IntPtr array_out, IntPtr array_in);
    }
}

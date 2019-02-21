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
	public static class AFIndex
	{
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_index(out IntPtr array_out, IntPtr array_in, uint ndims, [In] af_seq[] index);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_lookup(out IntPtr array_out, IntPtr array_in, IntPtr array_indices, uint dim);

		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_assign_seq(ref IntPtr array_out, IntPtr array_lhs, uint ndims, [In] af_seq[] indices, IntPtr array_rhs);

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_index_gen(out IntPtr array_out, IntPtr array_in, long dim_ndims, ???const af_index_t* indices???); */

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_assign_gen(out IntPtr array_out, IntPtr array_lhs, long dim_ndims, ???const af_index_t* indices???, IntPtr array_rhs); */

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_create_indexers(???af_index_t** indexers???); */

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_set_array_indexer(out ???af_index_t??? indexer, IntPtr array_idx, long dim_dim); */

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_set_seq_indexer(out ???af_index_t??? indexer, [In] af_seq[] idx, long dim_dim, bool is_batch); */

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_set_seq_param_indexer(out ???af_index_t??? indexer, double begin, double end, double step, long dim_dim, bool is_batch); */

		/* not yet supported:
		[DllImport(af_config.dll, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
		public static extern af_err af_release_indexers(out ???af_index_t??? indexers); */
	}
}

using System.Runtime.InteropServices;
using OpHandle = System.IntPtr;
using nn_uint = System.UInt32;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.Interop
{

    internal sealed partial class NativeMethods
    {

        #region Methods

        /// <summary>
        /// return str message of the last error
        /// <para>all function in this file will return 0 when success and -1 when an error occured, <see cref="NNGetLastError"/> can be called to retrieve the error</para>
        /// <para>this function is threadsafe and can be called by different thread</para>
        /// </summary>
        /// <returns>error info</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern OpHandle NNGetLastError();
        
        /// <summary>
        ///  Get operator handle given name.
        /// </summary>
        /// <param name="op_name">The name of the operator.</param>
        /// <param name="op_out">The returnning op handle.</param>
        /// <returns></returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int NNGetOpHandle(OpHandle op_name, out OpHandle op_out);

        /// <summary>
        /// list all the available operator names, include entries.
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output operator name array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int NNListAllOpNames(out nn_uint out_size, out OpHandle out_array);

        #endregion

    }

}

using System;
using System.Runtime.InteropServices;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.Interop
{

    internal sealed partial class NativeMethods
    {

        #region Methods
        
        [DllImport(CLibrary, CallingConvention = CallingConvention)]
        public static extern IntPtr memcpy(uint[] dest, IntPtr src, uint count);

        [DllImport(CLibrary, CallingConvention = CallingConvention)]
        public static extern IntPtr memcpy(float[] dest, IntPtr src, uint count);

        #endregion

    }

}

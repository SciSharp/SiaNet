using System.Runtime.InteropServices;

namespace SiaNet.Backend.MxNetLib.Interop
{

    internal sealed partial class NativeMethods
    {

        #region Constants

        public const int OK = 0;

        public const int Error = -1;

        public const int TRUE = 1;

        public const int FALSE = 0;

        #endregion

#if LINUX
        public const string NativeLibrary = "libmxnet.so";

        public const string CLibrary = "libc.so";

        public const CallingConvention CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl;
#else
        public const string NativeLibrary = @"libmxnet.dll"; 

        public const string CLibrary = "msvcrt.dll";

        public const CallingConvention CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl;
#endif

    }

}

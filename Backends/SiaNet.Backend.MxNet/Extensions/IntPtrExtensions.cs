using SiaNet.Backend.MxNetLib.Interop;

namespace SiaNet.Backend.MxNetLib.Extensions
{

    internal static class NatoveMethodsExtensions
    {

        public static int ToInt32(this bool source)
        {
            return source ? NativeMethods.TRUE : NativeMethods.FALSE;
        }

    }

}

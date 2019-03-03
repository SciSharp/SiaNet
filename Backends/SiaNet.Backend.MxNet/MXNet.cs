using SiaNet.Backend.MxNetLib.Interop;

// ReSharper disable once CheckNamespace
namespace SiaNet.Backend.MxNetLib
{

    public sealed partial class MXNet
    {

        #region Methods

        public static void MXNotifyShutdown()
        {
            Logging.CHECK_EQ(NativeMethods.MXNotifyShutdown(), NativeMethods.OK);
        }

        #endregion

    }

}

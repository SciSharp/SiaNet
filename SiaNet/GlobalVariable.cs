using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public class Global
    {
        public static IBackend Backend { get; set; }

        internal static ActivationFunc ActFunc = null;

        internal static bool UseCudnn { get; set; }

        public static void UseEngine(IBackend backend, DeviceType deviceType, bool cudnn = false)
        {
            Backend = backend;
            ActFunc = backend.GetActFunc();
            if(cudnn && deviceType != DeviceType.CUDA)
            {
                throw new ArgumentException("CuDnn work with CUDA device type");
            }

            UseCudnn = cudnn;
            Backend.SetDevice(deviceType);
        }
    }
}

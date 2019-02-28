using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace SiaNet
{
    public class Global
    {
        private static IBackend _backend;

        public static IBackend CurrentBackend
        {
            get
            {
                if(_backend == null)
                {
                    throw new NullReferenceException("Invoke Global.UseEngine() function first to setup the backend and device.");
                }

                return _backend;
            }
        }

        internal static ActivationFunc ActFunc = null;

        internal static bool UseCudnn { get; set; }

        public static void UseEngine(IBackend backend, DeviceType deviceType, bool cudnn = false)
        {
            _backend = backend;
            ActFunc = backend.GetActFunc();
            if(cudnn && deviceType != DeviceType.CUDA)
            {
                throw new ArgumentException("CuDnn work with CUDA device type");
            }

            UseCudnn = cudnn;
            _backend.SetDevice(deviceType);
        }
    }
}

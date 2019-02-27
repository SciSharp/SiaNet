using SiaNet.Backend.ArrayFire;
using SiaNet.Backend.TensorSharp;
using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet
{
    public class Global
    {
        public static IBackend backend = null;

        public static ActivationFunc ActFunc = null;

        private static SiaNetBackend BackendType { get; set; } = SiaNetBackend.ArrayFire;

        public static IBackend Backend
        {
            get
            {
                if(backend == null)
                {
                    switch (BackendType)
                    {
                        case SiaNetBackend.TensorSharp:
                            backend = new TensorSharpBackend();
                            ActFunc = new TensorSharpActivations(backend);
                            break;
                        case SiaNetBackend.ArrayFire:
                            backend = new ArrayFireBackend();
                            ActFunc = new ArrayFireActivations(backend);
                            break;
                        case SiaNetBackend.CNTK:
                            throw new NotImplementedException();
                        case SiaNetBackend.Tensorflow:
                            throw new NotImplementedException();
                        case SiaNetBackend.MxNet:
                            throw new NotImplementedException();
                        default:
                            break;
                    }
                    
                }

                return backend;
            }
        }

        internal static bool UseCudnn { get; set; }

        internal static int contextId = 0;

        public static void UseDevice(DeviceType deviceType, bool cudnn = false)
        {
            if(cudnn && deviceType != DeviceType.CUDA)
            {
                throw new ArgumentException("CuDnn work with CUDA device type");
            }

            UseCudnn = cudnn;
            Backend.SetDevice(deviceType);
        }

        public static void SetBackend(SiaNetBackend backend)
        {
            BackendType = backend;
        }
    }
}

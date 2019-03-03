using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.MxNetLib
{
    public class DeviceManager
    {
        public static Context Current { get; set; } = Context.Cpu();

        public static bool IsCuda { get; set; }

        public static void SetBackend(Backend deviceType, int gpuId = 0)
        {
            switch (deviceType)
            {
                case Backend.CPU:
                    Current = Context.Cpu();
                    break;
                case Backend.CUDA:
                    Current = Context.Gpu(gpuId);
                    IsCuda = true;
                    break;
                default:
                    break;
            }
        }
    }

    public enum Backend
    {
        CPU,
        CUDA
    }
}

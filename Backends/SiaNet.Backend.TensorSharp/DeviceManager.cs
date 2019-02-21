using SiaNet.Backend.TensorSharp.Cpu;
using SiaNet.Backend.TensorSharp.CUDA;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.TensorSharp
{
    public class DeviceManager
    {
        public static IAllocator Current { get; set; } = new CpuAllocator();

        public static bool IsCuda { get; set; }

        public static void SetBackend(Backend deviceType, int gpuId = 0)
        {
            switch (deviceType)
            {
                case Backend.CPU:
                    Current = new CpuAllocator();
                    break;
                case Backend.CUDA:
                    var cudaContext = new TSCudaContext();
                    cudaContext.Precompile(Console.Write);
                    cudaContext.CleanUnusedPTX();
                    Current = new CudaAllocator(cudaContext, gpuId);
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

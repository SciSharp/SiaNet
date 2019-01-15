using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.CUDA;

namespace SiaNet
{
    public class Global
    {
        public static IAllocator Device = new TensorSharp.Cpu.CpuAllocator();

        internal static bool UseCudnn { get; set; }

        internal static bool UseCuda { get; set; }

        public static void UseGpu(int gpuId = 0, bool cudnn = false)
        {
            SetNewContext(gpuId, true);
            UseCudnn = cudnn;
            UseCuda = true;
        }

        public static void SetNewContext(int gpuId = 0, bool compile = false)
        {
            var cudaContext = new TSCudaContext();
            if (compile)
            {
                cudaContext.Precompile(Console.Write);
                cudaContext.CleanUnusedPTX();
            }

            Device = new CudaAllocator(cudaContext, gpuId);
        }
    }
}

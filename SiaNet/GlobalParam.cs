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

        public static bool UseCudnn { get; set; }

        public static void UseGpu(int gpuId = 0, bool cudnn = false)
        {
            var cudaContext = new TSCudaContext();
            cudaContext.Precompile(Console.Write);
            cudaContext.CleanUnusedPTX();
            Device = new CudaAllocator(cudaContext, gpuId);
            UseCudnn = cudnn;
        }
    }
}

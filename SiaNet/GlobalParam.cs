using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.CUDA;

namespace SiaNet
{
    public class Global
    {
        public static Config Configuration { get; set; } = Config.GetConfig();

        private static IAllocator device;

        public static IAllocator Device
        {
            get
            {
                if (device == null)
                {
                    if (Configuration.UseGpu)
                    {
                        UseGpu(0);
                    }
                    else
                    {
                        device = new TensorSharp.Cpu.CpuAllocator();
                    }
                }

                return device;
            }
        }

        public static void UseGpu(int gpuId = 0)
        {
            var cudaContext = new TSCudaContext();
            cudaContext.Precompile(Console.Write);
            cudaContext.CleanUnusedPTX();
            device = new CudaAllocator(cudaContext, gpuId);
        }
    }
}

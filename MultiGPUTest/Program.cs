using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using SiaNet;
using TensorSharp;
using TensorSharp.CUDA;

namespace MultiGPUTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseGpu();
            Stopwatch sw = new Stopwatch();
            sw.Start();
            List<Thread> workers = new List<Thread>();
            for (int i = 0; i < 25; i++)
            {
                Thread t = new Thread(new ThreadStart(RunTask));
                t.Start();
                workers.Add(t);
            }

            foreach (var item in workers)
            {
                item.Join();
            }

            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds);
            Console.ReadLine();
        }

        private static void RunTask()
        {
            var cudaContext = new TSCudaContext();
            //cudaContext.Precompile();
            //cudaContext.CleanUnusedPTX();
            IAllocator allocator = new CudaAllocator(cudaContext, 0);

            Tensor a = new Tensor(allocator, DType.Float32, 3000, 3000);
            Tensor b = new Tensor(allocator, DType.Float32, 3000, 3000);

            TOps.RandomUniform(a, new SeedSource(), -10, 10);
            TOps.RandomUniform(b, new SeedSource(), -10, 10);

            var c = TOps.Dot(a, b) + b;
        }
    }
}

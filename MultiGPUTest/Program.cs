using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using SiaNet;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace MultiGPUTest
{
    class Program
    {
        static IAllocator cpu = new CpuAllocator();
        static Tensor x = null;
        static int current = -30;

        static void Main(string[] args)
        {
            Global.UseGpu();
            
            Stopwatch sw = new Stopwatch();
            x = new Tensor(cpu, DType.Float32, 6000, 30);
            TOps.RandomUniform(x, new SeedSource(), -20, 30);
            sw.Start();
            List<Thread> workers = new List<Thread>();
            int running = 4;
            ThreadPool.SetMaxThreads(4, 4);
            AutoResetEvent done = new AutoResetEvent(false);
            while (Next())
            {
                var data = GetBatch(current, 30);
                ThreadPool.QueueUserWorkItem(RunTask, data);
                Console.WriteLine(running);
                if (0 >= Interlocked.Decrement(ref running))
                    done.Set();
            }

            done.WaitOne();

            foreach (var item in workers)
            {
                item.Join();
            }

            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds);
            Console.ReadLine();
        }

        private static Tensor GetBatch(int start, int size, int axis = 0)
        {
            if (start + size <= x.Shape[0])
            {
                return x.Narrow(axis, start, size);
            }
            else
            {
                return x.Narrow(axis, start, x.Shape[0] - start);
            }
        }

        static bool Next()
        {
            current += 30;
            return current < x.Shape[0];
        }

        private static void RunTask(object state)
        {
            Tensor data_cpu = (Tensor)state;
            var cudaContext = new TSCudaContext();
            IAllocator allocator = new CudaAllocator(cudaContext, 0);
            Tensor data = new Tensor(allocator, DType.Float32, data_cpu.Shape);
            TOps.Copy(data, data_cpu);
            Tensor a = new Tensor(allocator, DType.Float32, 30, 3000);
            Tensor b = new Tensor(allocator, DType.Float32, 30, 3000);

            TOps.RandomUniform(a, new SeedSource(), -10, 10);
            TOps.RandomUniform(b, new SeedSource(), -10, 10);

            var c = TOps.Dot(data, a) + b;
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }
    }
}

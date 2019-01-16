using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
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
        static int batch = 300;
        static int current = -batch;
        static int contextCount = 0;

        static List<IAllocator> contextList = new List<IAllocator>();
        static void Main(string[] args)
        {
            Global.UseGpu();
            
            Stopwatch sw = new Stopwatch();
            x = new Tensor(cpu, DType.Float32, 20000, 3000);
            TOps.RandomUniform(x, new SeedSource(), -20, 30);
            sw.Start();
            
            ParallelOptions parallelOptions = new ParallelOptions() { MaxDegreeOfParallelism = 4 };

            //for (int i = 0; i < 4; i++)
            //{
            //    var cudaContext = new TSCudaContext();
            //    IAllocator allocator = new CudaAllocator(cudaContext, 0);
            //    contextList.Add(allocator);
            //}

            List<Task> taskList = new List<Task>();
            for (int i = 0; i < 100; i++)
            {
                var task = Task.Factory.StartNew(() => { RunEpoch(i); });
                taskList.Add(task);
            }

            Task.WaitAll(taskList.ToArray());

            //var nums = Enumerable.Range(1, 100);
            //var parallelQuery = from num in nums.AsParallel().WithDegreeOfParallelism(2).WithExecutionMode(ParallelExecutionMode.ForceParallelism)
            //                    select RunEpoch(num);

            //var parallelQuery = from num in nums
            //                    select RunEpoch(num);

            //foreach (var item in parallelQuery)
            //{
            //    Console.WriteLine("Epoch: " + item);
            //}

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
            current += batch;
            return current < x.Shape[0];
        }

        private static int RunEpoch(int i)
        {
            //int contextId = i < 4 ? i : i % 4;

            //Console.WriteLine(contextId);
            Global.SetNewContext();
            while (Next())
            {
                var data = GetBatch(current, batch);
                RunTask(Global.Device, data);
            }

            Console.WriteLine("Epoch: " + i);
            return i;
        }

        private static void RunTask(IAllocator allocator, Tensor data_cpu)
        {
            Tensor data = new Tensor(allocator, DType.Float32, data_cpu.Shape);
            TOps.Copy(data, data_cpu);

            Tensor a = new Tensor(allocator, DType.Float32, 3000, batch);
            Tensor b = new Tensor(allocator, DType.Float32, 3000, batch);

            TOps.RandomUniform(a, new SeedSource(), -10, 10);
            TOps.RandomUniform(b, new SeedSource(), -10, 10);

            var c = TOps.Dot(data, a) + b;
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }
    }
}

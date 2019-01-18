using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SiaNet;
using TensorSharp;
using TensorSharp.Expression;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace MultiGPUTest
{
    class Program
    {
        //static IAllocator cpu = new CpuAllocator();
        static Tensor x = null;
        static int batch = 128;
        static int current = -batch;
        static int contextCount = 0;
        static Variable a = null;
        static Variable b = null;
        static List<IAllocator> contextList = new List<IAllocator>();
        static void Main(string[] args)
        {
            Global.UseGpu();
            
            Stopwatch sw = new Stopwatch();
            x = new Tensor(Global.Device, DType.Float32, 6000, 3000);
            TOps.RandomUniform(x, new SeedSource(), -20, 30);
            sw.Start();
            

            List<Task> taskList = new List<Task>();
            var numEpoches = Enumerable.Range(0, 5);
            Parallel.ForEach(numEpoches, new ParallelOptions() { MaxDegreeOfParallelism = 2 }, (i) => {
                RunEpoch(i);
            });
            //for (int i = 0; i < 5; i++)
            //{
            //    var task = Task.Factory.StartNew(() =>
            //    {
            //        RunEpoch(i);
            //    });

            //    taskList.Add(task);
            //    //RunEpoch(i);
            //}

            //Task.WaitAll(taskList.ToArray());

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

        private static Tensor GetIndices(int count, params long[] shape)
        {
            List<float> data = new List<float>();
            for (int i = 0; i < count; i++)
            {
                var t = Tensor.Constant(i, Global.Device, DType.Float32, shape);
                data.AddRange(t.ToArray().Cast<float>());
            }

            var result = Tensor.FromArray(Global.Device, data.ToArray());
            return result.Reshape(-1, shape[1]);
        }

        private static int RunEpoch(int i)
        {
            //int contextId = i < 4 ? i : i % 4;

            //Console.WriteLine(contextId);
            Global.SetNewContext();
            current = -batch;
            a = Variable.RandomUniform(new SeedSource(), -10, 10, Global.Device, DType.Float32, 3000, batch);
            b = Variable.RandomUniform(new SeedSource(), -10, 10, Global.Device, DType.Float32, 3000, batch);

            while (Next())
            {
                var data = GetBatch(current, batch);
                var res = RunTask(Global.Device, data);
                //res = res.TVar().Gather(0, indices).Evaluate();
                //Console.WriteLine(TOps.MeanF(res));
            }

            Console.WriteLine("Epoch: " + i);
            return i;
        }

        private static Tensor RunTask(IAllocator allocator, Variable data)
        {
            var c = VarOps.Square(VarOps.Dot(data, a) + b) + VarOps.Square(VarOps.Dot(data, a) + b) + VarOps.Square(VarOps.Dot(data, a) + b) + VarOps.Square(VarOps.Dot(data, a) + b) + VarOps.Square(VarOps.Dot(data, a) + b);

            return c.Evaluate();

            //Tensor a = Variable.RandomUniform(new SeedSource(), -10, 10, allocator, DType.Float32, 3000, batch).Evaluate();
            //Tensor b = Variable.RandomUniform(new SeedSource(), -10, 10, allocator, DType.Float32, 3000, batch).Evaluate();

            //var c = TOps.Square(TOps.Dot(data.Evaluate(), a) + b);
        }
    }
}

using SiaNet;
using SiaNet.Engine;
using SiaNet.Initializers;
using System;

namespace BackendTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseEngine(SiaNet.Backend.TensorFlowLib.SiaNetBackend.Instance, DeviceType.CPU);
            var K = Global.CurrentBackend;

            var x = K.RandomNormal(new long[] { 10, 3 }, 0, 1);
            var x_shape = x.Shape;
            var sliced = K.SliceRows(x, 0, 2).ToArray();
            var a = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 3,3 });
            var b = K.CreateVariable(new float[] { 5 }, new long[] { 1, 1 });
            var shape = a.Shape;
            var c = a + b;

            //var d = K.CreateVariable(new float[] { 1 }, new long[] { 1, 1 });
            //c = c + d;
            c.Print();
            Console.ReadLine();
        }
    }
}

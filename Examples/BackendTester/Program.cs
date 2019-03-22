using SiaNet;
using SiaNet.Engine;
using SiaNet.Initializers;
using System;
using System.Linq;
using System.Numerics.Tensors;

namespace BackendTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseEngine(SiaNet.Backend.MxNetLib.SiaNetBackend.Instance, DeviceType.CPU);
            var K = Global.CurrentBackend;

            var constData = K.Constant(1, new long[] { 3, 3 });
            var a = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 3, 3 });
            var shape = a.Shape;
            var c = a + 5;

            //var d = K.CreateVariable(new float[] { 1 }, new long[] { 1, 1 });
            //c = c + d;
            c.Print();
            Console.ReadLine();

        }
    }
}

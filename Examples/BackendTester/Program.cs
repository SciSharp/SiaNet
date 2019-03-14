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
            Global.UseEngine(SiaNet.Backend.CNTKLib.SiaNetBackend.Instance, DeviceType.CPU);
            var K = Global.CurrentBackend;

            var a = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 9, 1 });
            var b = K.CreateVariable(new float[] { 5 }, new long[] { 1, 1 });

            var c = a + b;
            c.Print();
            Console.ReadLine();
        }
    }
}

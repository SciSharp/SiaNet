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
            Global.UseEngine(SiaNet.Backend.ArrayFire.SiaNetBackend.Instance, DeviceType.CPU);
            var K = Global.CurrentBackend;
            
            var a = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 3,3 });
            var b = K.CreateVariable(new float[] { 5 }, new long[] { 1, 1 });

            var f = K.Sum(a);

            //var c = a + b;

            //var d = K.CreateVariable(new float[] { 1 }, new long[] { 1, 1 });
            //c = c + d;

            Console.ReadLine();
        }
    }
}

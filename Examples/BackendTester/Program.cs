using NumSharp.Core;
using SiaNet;
using SiaNet.Engine;
using SiaNet.Initializers;
using System;
using System.Linq;

namespace BackendTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseEngine(SiaNet.Backend.CNTKLib.SiaNetBackend.Instance, DeviceType.CPU);
            var K = Global.CurrentBackend;

            var a = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 6, 3 });
            a.Print();

            var sliced = K.SliceRows(a, 1, 2);
            
            //var d = K.CreateVariable(new float[] { 1 }, new long[] { 1, 1 });
            //c = c + d;

            Console.ReadLine();

        }
    }
}

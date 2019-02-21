using SiaNet;
using SiaNet.Backend.TensorSharp;
using SiaNet.Data;
using SiaNet.Engine;
using System;

namespace BackendTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.SetBackend(SiaNetBackend.ArrayFire);
            var K = Global.Backend;
            var x = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 3, 3 });

            K.Print(x.SliceCols(2, 2));
        }
    }
}

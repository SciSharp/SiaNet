using SiaNet;
using SiaNet.Backend.TensorSharp;
using SiaNet.Engine;
using SiaNet.Initializers;
using System;

namespace BackendTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Global.UseEngine(TensorSharpBackend.Instance, DeviceType.CPU);
            Constant init = new Constant(3);
            var tensor = init.Operator(3, 3);
        }
    }
}

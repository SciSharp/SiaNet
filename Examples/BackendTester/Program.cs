using CNTK;
using NumSharp;
using SiaNet;
using SiaNet.Engine;
using SiaNet.Initializers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BackendTester
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] shape = new int[] { 6, 3 };
            float[] data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            NDArrayView array = new NDArrayView(shape, data, DeviceDescriptor.CPUDevice);
            Variable variable = new Variable(shape, VariableKind.Parameter, CNTK.DataType.Float, array, false, new AxisVector(), false, "", "");
            var slicedData = CNTKLib.Slice(variable, AxisVector.Repeat(new Axis(0), 1), IntVector.Repeat(1, 1), IntVector.Repeat(3, 1));
            var resultArray = GetArray(slicedData);


            Global.UseEngine(SiaNet.Backend.CNTKLib.SiaNetBackend.Instance, DeviceType.CPU);
            var K = Global.CurrentBackend;

            var a = K.CreateVariable(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new long[] { 6, 3 });
            a.Print();

            var sliced = K.SliceRows(a, 1, 2);
            
            //var d = K.CreateVariable(new float[] { 1 }, new long[] { 1, 1 });
            //c = c + d;

            Console.ReadLine();

        }

        public static Array GetArray(Variable xvar)
        {
            Value v = null;
            if (xvar.IsOutput)
            {
                var f = xvar.ToFunction();

                var plist = f.Parameters();
                Dictionary<Variable, Value> inputs = new Dictionary<Variable, Value>();
                Dictionary<Variable, Value> outputs = new Dictionary<Variable, Value>()
                {
                    { f, null}
                };

                f.Evaluate(inputs, outputs, DeviceDescriptor.CPUDevice);
                v = outputs.FirstOrDefault().Value;
            }
            else
            {
                v = new Value(xvar.GetValue());
            }

            return v.GetDenseData<float>(xvar)[0].ToArray();
        }
    }
}

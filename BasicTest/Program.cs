#define CUDA10

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using SiaNet;
using SiaNet.Data;
using SiaNet.Initializers;
using SiaNet.Layers;
using SiaNet.Layers.Activations;
using TensorSharp.Expression;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Zeros = SiaNet.Initializers.Zeros;

namespace Examples
{
    class BasicTest
    {
        static IAllocator device = new CpuAllocator();

        static void Main(string[] args)
        {
            //TestDense();
            //TestAct();

            TestLoss();
        }

        private static void TestDense()
        {
            Tensor tensor = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            tensor = tensor.Reshape(3, -1);
            Dense d = new Dense(6, ActivationType.Linear, new Ones(), null, null, true, new Ones());
            d.Forward(Variable.Create(tensor));
            d.Output.Print();

            d.Backward(d.Output);
            d.Input.Grad.Print();
            d.Params["w"].Grad.Print();
            d.Params["b"].Grad.Print();
        }
       
        private static void TestAct()
        {
            Tensor tensor = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            tensor = tensor.Reshape(3, -1);

            var act = new Selu();
            act.Forward(Variable.Create(tensor));
            act.Output.Print();

            act.Backward(act.Output);
            act.Input.Grad.Print();
        }

        private static void TestLoss()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { 9, 8, 7, 6, 5, 4, 3, 2, 1 });
            preds = preds.Reshape(3, -1);

            Tensor labels = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            labels = labels.Reshape(3, -1);

            var loss = new SiaNet.Losses.CosineProximity();
            var l = loss.Call(preds, labels);
            var data = l.ToArray();

            var grad = loss.CalcGrad(preds, labels);
            grad.Print();
        }
    }
}

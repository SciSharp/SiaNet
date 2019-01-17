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
using BasicTest;

namespace Examples
{
    class BasicTest
    {
        static IAllocator device = new CpuAllocator();

        static void Main(string[] args)
        {
            Global.UseGpu();
            MNIST.Run();

            //TestDense();
            //TestAct();

            //TestLoss();
            Console.ReadLine();
        }

        private static void TestDense()
        {
            Tensor tensor = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            tensor = tensor.Reshape(3, -1);
            tensor.ToParameter();
            Dense d = new Dense(6, ActivationType.Linear, new Ones(), null, null, true, new Ones());
            d.Forward(Parameter.Create(tensor));
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
            act.Forward(Parameter.Create(tensor));
            act.Output.Print();

            act.Backward(act.Output);
            act.Input.Grad.Print();
        }
         
        private static void TestLoss()
        {
            Tensor preds = Tensor.FromArray(Global.Device, new float[] { 0.7f, 0.1f, 0.1f, 0.4f, 0.55f, 0.05f, 0.9f, 0.01f, 0.09f });
            preds = preds.Reshape(-1, 1);
            (preds > 0.25f).Print();
            Tensor labels = Tensor.FromArray(Global.Device, new float[] { 1, 0, 0, 0, 0, 1, 0, 1, 0 });
            labels = labels.Reshape(-1, 1);

            var loss = new SiaNet.Losses.BinaryCrossentropy();
            var l = loss.Call(preds, labels);
            l.Print();

            var grad = loss.CalcGrad(preds, labels);
            grad.Print();
        }
    }
}

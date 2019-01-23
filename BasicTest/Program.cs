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
            //SoftmaxAct();
            Console.ReadLine();
        }

        private static void SoftmaxAct()
        {
            Tensor x = Tensor.FromArray(Global.Device, new float[] { 1, -2, 3, 4, -5, 6, 7, -8, 9 });
            x = x.Reshape(3, -1);
            (x - TOps.Max(x, -1)).Print();
            var y = TOps.Softmax(x);
            y.Print();

            var grad = x - y / TOps.Sum(y, -1);
            grad.Print();
        }
       
        private static void TestAct()
        {
            Tensor tensor = Tensor.FromArray(Global.Device, new float[] { -1, 2, 3, -4, 5, 6, 7, -8, 9 });
            tensor = tensor.Reshape(3, -1);

            var act = new Selu();
            act.Forward(tensor);
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

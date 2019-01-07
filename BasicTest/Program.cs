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
            //Global.UseGpu();
            //SetConstValue();
            //Elu();
            //TestSoftmax();
            //MNIST.Run();
            //TestActivations.Run();

            //MaxImpl();
            //ToArrayTest();
            //FlattenTest();
            TestConv1d();
            //TestParallel();
        }

        private static void TestDense()
        {
            Tensor tensor = new Tensor(device, DType.Float32, 4, 2);
            RandomUniform init = new RandomUniform(-10, 10);
            int dim = 4;
            tensor = init.Operator(tensor);
            Console.WriteLine(tensor.Format());
            Dense d = new Dense(4, ActivationType.ReLU, new GlorotUniform(), null, null, true, new Zeros());
            d.Forward(Variable.Create(tensor));
            
            Console.WriteLine(d.Output.Format());

        }
        private static void SetConstValue()
        {
            float value = 0.32f;
            
            Tensor tensor = new Tensor(device, DType.Float32, 4, 2);
            //tensor = TVar.Fill(value, device, DType.Float32, tensor.Sizes).Evaluate();
            Ops.Fill(tensor, 0.33f);

            Console.WriteLine(tensor.Format());
        }

        private static void TestInit()
        {
            Tensor tensor = new Tensor(device, DType.Float32, 4, 2);
            Ones glorotUniform = new Ones();

            tensor = glorotUniform.Operator(tensor);
            Console.WriteLine(tensor.Format());
        }

        private static void TestArgMax()
        {
            Tensor tensor = new Tensor(device, DType.Float32, 5, 10);

            RandomUniform init = new RandomUniform(0, 1);

            tensor = init.Operator(tensor);

            Console.WriteLine(tensor.Format());
            Console.WriteLine(tensor.TVar().Argmax(1).Evaluate().Format());
        }

        private static void TestGreaterThan()
        {
            Tensor tensor = new Tensor(device, DType.Float32, 5, 2);

            RandomUniform init = new RandomUniform(-1, 1);

            tensor = init.Operator(tensor);

            Console.WriteLine(tensor.Format());

            Ops.GreaterThan(tensor, tensor, 0);

            Console.WriteLine(tensor.Format());
        }

        private static void Relu()
        {
            Tensor tensor = new Tensor(device, DType.Float32, 5, 2);

            RandomUniform init = new RandomUniform(-1, 1);

            tensor = init.Operator(tensor);

            Console.WriteLine(tensor.Format());

            var keepElements = tensor.TVar() > 0;
            tensor = (tensor.TVar().CMul(keepElements) + (1 - keepElements) * 0).Evaluate();

            tensor = tensor.TVar().CMul(tensor.TVar() > 0).Evaluate();

            Console.WriteLine(tensor.Format());
        }

        private static void Elu()
        {
            Tensor tensor = new Tensor(device, DType.Float32, 5, 2);
            float alpha = 1;
            RandomUniform init = new RandomUniform(-1, 1);

            tensor = init.Operator(tensor);

            Console.WriteLine(tensor.Format());

            tensor = _elu(tensor).Evaluate();
           
            //var keepElements = tensor.TVar() >= 0;
            //TVar d = (alpha * (tensor.TVar().Exp() - 1)).Evaluate();
            //tensor = (tensor.TVar().CMul(keepElements) + (1 - keepElements).CMul(d)).Evaluate();

            Console.WriteLine(tensor.Format());
        }

        private static TVar _elu(TVar input)
        {
            float alpha = 1;
            var keepElements = input >= 0;
            TVar d = (alpha * (input.Exp() - 1)).Evaluate();
            return (input.CMul(keepElements) + (1 - keepElements).CMul(d)).Evaluate();
        }

        private static void TestSoftmax()
        {
            Tensor tensor = TVar.FromArray(new float[] { 1, -2, 3, -4, 1, 2, -3, 4 }, Global.Device).Evaluate();
            tensor = tensor.View(2, 4);

            tensor.Print();
            Softmax s = new Softmax();
            s.Forward(Variable.Create(tensor));
            s.Output.Print();
            
        }

        private static void MaxImpl()
        {
            Tensor tensor1 = TVar.FromArray(new float[] { 1, 2, 3, 4, 1, 2, 3, 1 }, Global.Device).Evaluate();
            Tensor tensor2 = TVar.FromArray(new float[] { -1, 4, -3, 4, 6, 1, 2, -2 }, Global.Device).Evaluate();

            tensor1 = tensor1.View(4, 2);
            tensor2 = tensor2.View(4, 2);

            var t1 = (tensor1.TVar() >= tensor2.TVar());
            var t2 = (tensor2.TVar() > tensor1.TVar());

            (t1.CMul(tensor1) + t2.CMul(tensor2)).Print();
        }

        private static void ToArrayTest()
        {
            Tensor tensor = TVar.RandomNormal(new SeedSource(), 2, 1, Global.Device, DType.Float32, 100, 300).Evaluate();
            
            var arr = tensor.ToArray();
        }

        private static void FlattenTest()
        {
            Tensor tensor1 = TVar.FromArray(new float[] { 1, 2, 3, 4, 1, 2, 3, 1 }, Global.Device).Evaluate();

            tensor1 = tensor1.View(4, 2);

            var c = tensor1.ElementCount();

            tensor1.View(c, -1).Print();
        }

        private static void PadTest()
        {
            Tensor tensor1 = TVar.RandomUniform(new SeedSource(), 0, 10, Global.Device, DType.Float32, 2, 2, 2, 4).Evaluate();
            tensor1.Print();
            
            tensor1 = tensor1.Pad(2, 0);
            tensor1.Print();
        }

        private static void TestConv2d()
        {
            var t = Tensor.Arange(Global.Device, 1, 10, 3);
            Tensor tensor1 = TVar.FromArray(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, Global.Device).Evaluate();
            tensor1 = tensor1.View(1, 2, 3, 3);
            Conv2D conv2D = new Conv2D(3, new Tuple<uint, uint>(3, 3), kernalInitializer: new Ones());
            conv2D.Forward(Variable.Create(tensor1));
            conv2D.Output.Print();
        }

        private static void TestConv1d()
        {
            var t = Tensor.Arange(Global.Device, 1, 10, 3);
            Tensor tensor1 = TVar.FromArray(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, Global.Device).Evaluate();
            tensor1 = tensor1.View(2, 3, 3);
            Conv1D conv = new Conv1D(3, 3, kernalInitializer: new Ones());
            conv.Forward(Variable.Create(tensor1));
            conv.Output.Print();
        }

        private static void TestConv3d()
        {
            Tensor tensor1 = TVar.FromArray(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, Global.Device).Evaluate();
            tensor1 = tensor1.View(1, 2, 2, 3, 3);
            Conv3D conv = new Conv3D(3, new Tuple<uint, uint, uint>(2, 3, 3), kernalInitializer: new Ones());
            conv.Forward(Variable.Create(tensor1));
            conv.Output.Print();
        }

        private static void TestParallel()
        {
            Tensor a = Tensor.FromArray(Global.Device, new float[] { 2, 3, 1, 2 });
            Tensor b = Tensor.FromArray(Global.Device, new float[] { 1, 2, 2, 3, 1, 2 });

            //Tensor c = new Tensor(Global.Device, DType.Float32, 2, 3);

            a = a.View(2, 2);
            b = b.View(2, 3);

            a.Print();
            b.Print();

            var c = a * 2;

            //Ops.Dot(c, a, b);
            c.Print();

            Matrix<float> ma = Matrix<float>.Build.DenseOfArray(new float[,] { { 2, 3 }, { 1, 2 } });
            Matrix<float> mb = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 2 }, {3 , 1, 2 } });

            var mc = ma * mb;
            Console.WriteLine(mc.ToMatrixString());
        }
    }
}

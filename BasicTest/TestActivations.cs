using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using SiaNet;
using SiaNet.Layers;
using SiaNet.Layers.Activations;
using TensorSharp.Expression;

namespace BasicTest
{
    public class TestActivations
    {
        public static void Run()
        {
            Tensor data = new Tensor(Global.Device, DType.Float32, 3, 5);
            Tensor indices = new Tensor(Global.Device, DType.Float32, 3, 5);

            data.CopyFrom(new float[] { -1, 3, 5, -7, 9, 0, 2, -4, -6, 8, 1, 2, 3, 4, 5 });
            indices.CopyFrom(new float[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 });
            data.Print("Input");

            var m = data.TVar().Max(1);
            m.Print();

            RunAct(new Elu(), data);
            RunAct(new Exp(), data);
            //RunAct(new HardSigmoid(), data);
            //RunAct(new LeakyRelu(), data);
            RunAct(new Linear(), data);
            //RunAct(new PRelu(), data);
            RunAct(new Selu(), data);
            RunAct(new Sigmoid(), data);
            RunAct(new Softmax(), data);
            RunAct(new Softplus(), data);
            RunAct(new Softsign(), data);
            RunAct(new Tanh(), data);

            Console.ReadLine();
        }

        private static void RunAct(BaseLayer act, Tensor x)
        {
            act.Forward(Parameter.Create(x));
            act.Output.Print(act.Name);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Softmax : BaseLayer
    {
        public Softmax()
            : base("softmax")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = Softmax(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * Output * (1 - Output);

            //var grad = Tensor.Constant(0, Global.Device, DType.Float32, Input.Grad.Shape);

            //var enumi = Enumerable.Range(0, (int)Output.Shape[0]);
            //var enumj = Enumerable.Range(0, (int)Output.Shape[1]);

            //foreach (var i in enumi.AsParallel())
            //{
            //    foreach (var j in enumj.AsParallel())
            //    {
            //        float v = 0;
            //        if (i == j)
            //        {
            //            v = Output.GetElementAsFloat(i, j) * (1 - Output.GetElementAsFloat(i, j));
            //        }
            //        else
            //        {
            //            v = -1 * Output.GetElementAsFloat(i, j) * Output.GetElementAsFloat(i, j);
            //        }

            //        grad.SetElementAsFloat(v, i, j);
            //    }
            //}

            //Input.Grad = outputgrad * grad;
        }
    }
}

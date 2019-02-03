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
            //Input.Grad = -1 * outputgrad * Output * Output;

            var scattered = Scatter(Output, 0, Output);
            var dx = Output * outputgrad;
            var s = outputgrad * Sum(dx, -1);
            Input.Grad = dx - (Output * s);
            

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
            //        elsex 
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

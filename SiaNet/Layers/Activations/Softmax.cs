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
            //var s = Output.Reshape(-1, 1);
            //var d = Diag(s) - Dot(s, s.Transpose());
            //Input.Grad = outputgrad * Sum(d, -1).Reshape(Input.Data.Shape);
        }
    }
}

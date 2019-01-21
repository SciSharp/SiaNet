using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Linear : BaseLayer
    {
        public Linear()
            : base("linear")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = x;
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}

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

        public override void Forward(Variable x)
        {
            Input = x;
            Output = x.Data;
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Input.Data;
        }
    }
}

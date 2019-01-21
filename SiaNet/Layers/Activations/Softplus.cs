using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Softplus : BaseLayer
    {
        public Softplus()
            : base("softplus")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = Softplus(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * (Exp(Input.Data) / (Exp(Input.Data) + 1));
        }
    }
}

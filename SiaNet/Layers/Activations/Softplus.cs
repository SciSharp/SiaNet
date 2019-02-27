using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            base.Forward(x);
            Output = K.Softplus(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * (K.Exp(Input.Data) / (K.Exp(Input.Data) + 1));
        }
    }
}

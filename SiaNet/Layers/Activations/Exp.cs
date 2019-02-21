using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Exp : BaseLayer
    {
        public Exp()
            : base("exp")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = K.Exp(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * Output;
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Tanh : BaseLayer
    {
        public Tanh()
            : base("tanh")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            Output = K.Tanh(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * (1 - K.Square(Output));
        }
    }
}

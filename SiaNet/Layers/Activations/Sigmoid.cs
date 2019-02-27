using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Sigmoid : BaseLayer
    {
        public Sigmoid()
            : base("sigmoid")
        {
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = K.Sigmoid(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * Output * (1 - Output);
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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
            base.Forward(x);
            Output = x;
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Softsign : BaseLayer
    {
        public Softsign()
            : base("softsign")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();

            Output = x / (K.Abs(x) + 1);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad / K.Square(K.Abs(Input.Data) + 1);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Softsign : BaseLayer
    {
        public Softsign()
            : base("softsign")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;

            Output = x.Data / (Abs(x.Data) + 1);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad / Square(Abs(Input.Data) + 1);
        }
    }
}

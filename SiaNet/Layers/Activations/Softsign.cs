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

            Output = Variable.Create((x.Data.TVar().CDiv(1 + x.Data.TVar().Abs()).Evaluate()));
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.TVar().CMul(1 / (1 + Output.Data.TVar().Abs())).Evaluate();
        }
    }
}

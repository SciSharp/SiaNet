using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Exp : BaseLayer
    {
        public Exp()
            : base("exp")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;
            Output = Variable.Create(x.Data.TVar().Exp().Evaluate());
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.TVar().CMul(Output.Data).Evaluate();
        }
    }
}

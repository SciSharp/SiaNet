using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Tanh : BaseLayer
    {
        public Tanh()
            : base("tanh")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;
            Output = Variable.Create(x.Data.TVar().Tanh().Evaluate());
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.TVar().CMul(1 - Output.Data.TVar().Pow(2)).Evaluate();
        }
    }
}

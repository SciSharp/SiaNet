using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Sigmoid : BaseLayer
    {
        public Sigmoid()
            : base("sigmoid")
        {
        }

        public override void Forward(Variable x)
        {
            Output = x.Data.TVar().Sigmoid().Evaluate();
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.TVar()
                .CMul(1 - Output.TVar())
                .CMul(Output)
                .Evaluate();
        }
    }
}

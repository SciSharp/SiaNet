using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Softplus : BaseLayer
    {
        public Softplus()
            : base("softplus")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;
            Output = Variable.Create(x.Data.TVar().Softplus().Evaluate());
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Ops.Exp(Input.Grad, Output.Data);
            Input.Grad = outputgrad.TVar().CMul(1 / (1 + Input.Grad.TVar())).Evaluate();
        }
    }
}

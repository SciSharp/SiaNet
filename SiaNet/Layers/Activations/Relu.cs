using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Relu : BaseLayer
    {
        public Relu()
            : base("relu")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;
            var keepElements = x.Data.TVar() > 0;
            Output = Variable.Create((x.Data.TVar().CMul(keepElements) + (1 - keepElements) * 0).Evaluate());
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.TVar().CMul(Output.Data.TVar() > 0).Evaluate();
        }
    }
}

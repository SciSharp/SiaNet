using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Elu : BaseLayer
    {
        private float _alpha;
        public Elu(float alpha = 1)
            : base("elu")
        {
            _alpha = alpha;
        }

        public override void Forward(Variable x)
        {
            Input = x;
            var keepElements = x.Data.TVar() > 0;
            var d = _alpha * (x.Data.TVar().Exp() - 1);
            Output = Variable.Create((x.Data.TVar().CMul(keepElements) + (1 - keepElements).CMul(d)).Evaluate());
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Output.Data.TVar() > 0;
            var d = _alpha * (Output.Data.TVar().Exp() - 1);
            Input.Grad = (outputgrad.TVar().CMul(keepElements) + (1 - keepElements).CMul(d)).Evaluate();
        }
    }
}

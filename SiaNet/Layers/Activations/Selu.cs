using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Selu : BaseLayer
    {
        private float alpha = 1.6732632423543772848170429916717f;
        private float scale = 1.0507009873554804934193349852946f;
        public Selu()
            : base("selu")
        {
        }

        public override void Forward(Variable x)
        {
            Input = x;

            var elu = new Elu(alpha);
            elu.Forward(x);
            Output = (elu.Output.TVar() * scale).Evaluate();
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Output.TVar() > 0;
            var d = scale * alpha * (Output.TVar().Exp());
            Input.Grad = (outputgrad.TVar().CMul(keepElements) + (1 - keepElements).CMul(d)).Evaluate();
        }
    }
}

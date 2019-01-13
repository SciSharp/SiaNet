using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Elu : BaseLayer
    {
        public float Alpha { get; set; }

        public Elu(float alpha = 1)
            : base("elu")
        {
            Alpha = alpha;
        }

        public override void Forward(Variable x)
        {
            Input = x;
            var keepElements = x.Data > 0;
            var keepElements_Exp = x.Data < 0;
            var d = Alpha * (Exp(Mul(x.Data, keepElements_Exp)) - 1);
            Output = Mul(x.Data, keepElements) + d;
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Input.Data > 0;
            var keepElements_Exp = Input.Data < 0;
            var d = Alpha * Exp(Mul(Input.Data, keepElements_Exp));
            Input.Grad = outputgrad * d;
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            var keepElements = x > 0;
            var keepElements_Exp = x < 0;
            var d = Alpha * (K.Exp(K.Mul(x, keepElements_Exp)) - 1);
            Output = K.Mul(x, keepElements) + d;
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Input.Data > 0;
            var keepElements_Exp = Input.Data < 0;
            var d = Alpha * K.Exp(K.Mul(Input.Data, keepElements_Exp));
            Input.Grad = outputgrad * d;
        }
    }
}

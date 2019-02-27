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
            base.Forward(x);
            Output = Global.ActFunc.EluForward(Alpha, x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.EluBackward(Alpha, Input.Data, outputgrad);
        }
    }
}

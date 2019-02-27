using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class LeakyRelu : BaseLayer
    {
        public float Alpha { get; set; }

        public LeakyRelu(float alpha = 0.3f)
            : base("leaky_relu")
        {
            Alpha = alpha;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = Global.ActFunc.LeakyReluForward(Alpha, x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.LeakyReluBackward(Alpha, Input.Data, outputgrad);
        }
    }
}

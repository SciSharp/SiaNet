using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

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

        public override void Forward(Variable x)
        {
            Input = x;
            var keepElements = x.Data >= 0;
            Output = x.Data * keepElements + (Alpha * x.Data * (1 - keepElements));
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Input.Data >= 0;
            Input.Grad = outputgrad * (keepElements + (Alpha * (1 - keepElements)));
        }
    }
}

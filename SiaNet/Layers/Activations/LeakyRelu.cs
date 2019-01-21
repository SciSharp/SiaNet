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

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            var keepElements = x >= 0;
            Output = x * keepElements + (Alpha * x * (1 - keepElements));
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Input.Data >= 0;
            Input.Grad = outputgrad * (keepElements + (Alpha * (1 - keepElements)));
        }
    }
}

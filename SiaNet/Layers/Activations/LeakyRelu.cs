using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class LeakyRelu : BaseLayer
    {
        private float _alpha;

        public LeakyRelu(float alpha = 0.3f)
            : base("leaky_relu")
        {
            _alpha = alpha;
        }

        public override void Forward(Variable data)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Tensor outputgrad)
        {
            
        }
    }
}

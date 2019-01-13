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
            var keepElements = x.Data > 0;
            Output = x.Data * keepElements + (1 - keepElements) * 0;
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Input.Data > 0;
            Input.Grad = outputgrad * (keepElements + (1 - keepElements) * 0);
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Relu : BaseLayer
    {
        public Relu()
            : base("relu")
        {
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            var keepElements = x > 0;
            Output = x * keepElements + (1 - keepElements) * 0;
        }

        public override void Backward(Tensor outputgrad)
        {
            var keepElements = Input.Data > 0;
            Input.Grad = outputgrad * (keepElements + (1 - keepElements) * 0);
        }
    }
}

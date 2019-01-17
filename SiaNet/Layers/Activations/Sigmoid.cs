using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class Sigmoid : BaseLayer
    {
        public Sigmoid()
            : base("sigmoid")
        {
        }

        public override void Forward(Parameter x)
        {
            Input = x;
            Output = Sigmoid(x.Data);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad * Output * (1 - Output);
        }
    }
}

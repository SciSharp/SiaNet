using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class PRelu : BaseLayer
    {
        public PRelu()
            : base("prelu")
        {
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

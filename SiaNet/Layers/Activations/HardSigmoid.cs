using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers.Activations
{
    public class HardSigmoid : BaseLayer
    {
        public HardSigmoid()
            : base("hard_sigmoid")
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

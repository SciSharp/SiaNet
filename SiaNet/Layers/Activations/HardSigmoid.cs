using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class HardSigmoid : BaseLayer
    {
        public HardSigmoid()
            : base("hard_sigmoid")
        {
        }

        public override void Forward(Tensor x)
        {
            throw new NotImplementedException();
            
        }

        public override void Backward(Tensor outputgrad)
        {
            throw new NotImplementedException();
        }
    }
}

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
            base.Forward(x);
            Output = Global.ActFunc.ReluForward(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.ReluBackward(Input.Data, outputgrad);
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Softmax : BaseLayer
    {
        public Softmax()
            : base("softmax")
        {
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = Global.ActFunc.SoftmaxForward(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = Global.ActFunc.SoftmaxBackward(Input.Data, outputgrad);
        }
    }
}

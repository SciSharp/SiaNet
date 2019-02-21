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
            Input = x.ToParameter();
            Output = K.Softmax(x);
        }

        public override void Backward(Tensor outputgrad)
        {
            var s = Output.Reshape(-1, 1);
            var d = K.Diag(s) - K.Dot(s, s.Transpose());
            Input.Grad = outputgrad * K.Sum(d, -1).Reshape(Input.Data.Shape);
        }
    }
}

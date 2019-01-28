using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Layers
{
    public class Flatten : BaseLayer
    {
        public Flatten()
             : base("flatten")
        {

        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();

            Output = x.Reshape(x.Shape[0], -1);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.Reshape(Input.Data.Shape);
        }
    }
}

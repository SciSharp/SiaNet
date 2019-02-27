using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class Permute : BaseLayer
    {
        public int[] Dims { get; set; }

        public Permute(params int[] dims)
            : base("permute")
        {
            Dims = dims;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = x.Transpose(Dims);
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad.Reshape(Input.Data.Shape);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

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

        public override void Forward(Variable x)
        {
            Input = x;

            Output = x.Data.Permute(Dims);
        }

        public override void Backward(Tensor outputgrad)
        {
        }
    }
}

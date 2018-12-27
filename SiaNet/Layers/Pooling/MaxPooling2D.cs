using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public class MaxPooling2D : BaseLayer
    {
        public Tuple<uint, uint> PoolSize { get; set; }

        public Tuple<uint, uint> Strides { get; set; }

        public uint? Padding { get; set; }

        public MaxPooling2D(Tuple<uint, uint> poolSize = null, Tuple<uint, uint> strides = null, uint? padding = null)
            : base("maxpooling2d")
        {
            PoolSize = poolSize ?? Tuple.Create<uint, uint>(2, 2);
            Strides = strides ?? poolSize;
            Padding = padding;
        }

        public override void Forward(Variable x)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Tensor outputgrad)
        {
            throw new NotImplementedException();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public class MaxPooling1D : BaseLayer
    {
        public uint PoolSize { get; set; }

        public uint Strides { get; set; }

        public uint? Padding { get; set; }

        public MaxPooling1D(uint poolSize = 2, uint? strides = null, uint? padding = null)
            : base("maxpooling1d")
        {
            PoolSize = poolSize;
            Strides = strides.HasValue ? padding.Value : poolSize;
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

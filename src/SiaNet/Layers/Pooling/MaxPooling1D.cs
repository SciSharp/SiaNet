using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Layers
{
    public class MaxPooling1D : BaseLayer, ILayer
    {
        public uint PoolSize { get; set; }

        public uint Strides { get; set; }

        public uint? Padding { get; set; }

        public MaxPooling1D(uint poolSize = 2, uint? strides = null, uint? padding = null)
            :base("maxpooling1d")
        {
            PoolSize = poolSize;
            Strides = strides.HasValue ? padding.Value : poolSize;
            Padding = padding;
        }

        public Symbol Build(Symbol x)
        {
            Shape pad = new Shape(); ;
            if (Padding.HasValue)
            {
                pad = new Shape(Padding.Value);
            }

            return Operators.Pooling(ID, x, new Shape(PoolSize), PoolingPoolType.Max, false, GlobalParam.UseCudnn, 
                                    PoolingPoolingConvention.Valid, new Shape(Strides), pad);
        }
    }
}

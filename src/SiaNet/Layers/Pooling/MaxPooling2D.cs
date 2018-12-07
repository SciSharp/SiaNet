using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Layers
{
    public class MaxPooling2D : BaseLayer, ILayer
    {
        public Tuple<uint, uint> PoolSize { get; set; }

        public Tuple<uint, uint> Strides { get; set; }

        public uint? Padding { get; set; }

        public MaxPooling2D(Tuple<uint, uint> poolSize = null, Tuple<uint, uint> strides = null, uint? padding = null)
            :base("maxpooling2d")
        {
            PoolSize = poolSize ?? Tuple.Create<uint, uint>(2, 2);
            Strides = strides ?? poolSize;
            Padding = padding;
        }

        public Symbol Build(Symbol x)
        {
            Shape pad = new Shape(); ;
            if (Padding.HasValue)
            {
                pad = new Shape(Padding.Value);
            }

            return Operators.Pooling(ID, x, new Shape(PoolSize.Item1, PoolSize.Item2), PoolingPoolType.Max, false, GlobalParam.UseCudnn, 
                                    PoolingPoolingConvention.Valid, new Shape(Strides.Item1, Strides.Item2), pad);
        }
    }
}

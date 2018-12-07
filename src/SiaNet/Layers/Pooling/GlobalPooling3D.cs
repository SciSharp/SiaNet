using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Layers
{
    public class GlobalPooling3D : BaseLayer, ILayer
    {
        public PoolingPoolType PoolingType { get; set; }

        public GlobalPooling3D(PoolingPoolType poolingType)
            :base("globalpooling3d")
        {
            PoolingType = poolingType;
        }

        public Symbol Build(Symbol x)
        {
            return Operators.Pooling(ID, x, new Shape(), PoolingType, true, GlobalParam.UseCudnn, 
                                    PoolingPoolingConvention.Valid, new Shape(), new Shape());
        }
    }
}

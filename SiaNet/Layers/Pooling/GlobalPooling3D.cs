using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Layers
{
    public class GlobalPooling3D : BaseLayer
    {
        public PoolingPoolType PoolingType { get; set; }

        public GlobalPooling3D(PoolingPoolType poolingType)
            : base("globalpooling3d")
        {
            PoolingType = poolingType;
        }

        public override void Forward(Parameter x)
        {
            Input = x;
            if (PoolingType == PoolingPoolType.Max)
            {
                Output = Max(x.Data, 2, 3, 4);
            }
            else
            {
                Output = Mean(x.Data, 2, 3, 4);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
        }
    }
}

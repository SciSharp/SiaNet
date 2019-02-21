using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

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

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            if (PoolingType == PoolingPoolType.Max)
            {
                Output = K.Max(x, 2, 3, 4);
            }
            else
            {
                Output = K.Mean(x, 2, 3, 4);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}

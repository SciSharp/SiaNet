using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class GlobalPooling2D : BaseLayer
    {
        public PoolingPoolType PoolingType { get; set; }

        public GlobalPooling2D(PoolingPoolType poolingType)
            : base("globalpooling2d")
        {
            PoolingType = poolingType;
        }

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            if (PoolingType == PoolingPoolType.Max)
            {
                Output = K.Max(x, 2, 3);
            }
            else
            {
                Output = K.Mean(x, 2, 3);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}

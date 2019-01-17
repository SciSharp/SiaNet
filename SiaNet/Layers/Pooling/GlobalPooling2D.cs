using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

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

        public override void Forward(Parameter x)
        {
            Input = x;
            if (PoolingType == PoolingPoolType.Max)
            {
                Output = Max(x.Data, 2, 3);
            }
            else
            {
                Output = Mean(x.Data, 2, 3);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
        }
    }
}

using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers
{
    public class GlobalPooling1D : BaseLayer
    {
        public PoolingPoolType PoolingType { get; set; }

        public GlobalPooling1D(PoolingPoolType poolingType)
            : base("globalpooling1d")
        {
            PoolingType = poolingType;
        }

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            if(PoolingType == PoolingPoolType.Max)
            {
                Output = K.Max(x, 2);
            }
            else
            {
                Output = K.Mean(x, 2);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}

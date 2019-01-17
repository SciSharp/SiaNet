using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

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

        public override void Forward(Parameter x)
        {
            Input = x;
            if(PoolingType == PoolingPoolType.Max)
            {
                Output = Max(x.Data, 2);
            }
            else
            {
                Output = Mean(x.Data, 2);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
        }
    }
}

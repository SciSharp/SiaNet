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

        public override void Forward(Tensor x)
        {
            Input = x.ToParameter();
            if (PoolingType == PoolingPoolType.Max)
            {
                Output = Max(x, 2, 3);
            }
            else
            {
                Output = Mean(x, 2, 3);
            }
        }

        public override void Backward(Tensor outputgrad)
        {
            Input.Grad = outputgrad;
        }
    }
}

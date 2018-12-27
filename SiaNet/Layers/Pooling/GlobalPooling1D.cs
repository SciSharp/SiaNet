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

        public override void Forward(Variable x)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Tensor outputgrad)
        {
            throw new NotImplementedException();
        }
    }
}

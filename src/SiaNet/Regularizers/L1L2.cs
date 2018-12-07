using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Regularizers
{
    public class L1L2 : BaseRegularizer
    {
        public L1L2(float l1=0.01f, float l2=0.01f)
            : base(l1, l2)
        {

        }

        public override NDArray Call(NDArray x)
        {
            NDArray regularizer = new NDArray();
            if(L1 > 0)
            {
                regularizer += NDArray.Sum(NDArray.Abs(x) * L1);
            }

            if (L2 > 0)
            {
                regularizer += NDArray.Sum(NDArray.Square(x) * L2);
            }

            return regularizer;
        }
    }
}

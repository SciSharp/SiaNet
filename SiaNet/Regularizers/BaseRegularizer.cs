using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Regularizers
{
    public abstract class BaseRegularizer : TOps
    {
        public string Name { get; set; }

        internal float L1 { get; set; }

        internal float L2 { get; set; }

        public BaseRegularizer(float l1 = 0.01f, float l2 = 0.01f)
        {
            L1 = l1;
            L2 = l2;
        }

        public abstract float Call(Tensor x);

        public abstract Tensor CalcGrad(Tensor x);
    }
}

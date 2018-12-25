using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Constraints
{
    public abstract class BaseConstraint
    {
        public abstract Tensor Call(Tensor w);
    }
}

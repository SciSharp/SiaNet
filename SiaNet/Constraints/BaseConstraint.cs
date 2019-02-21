using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Constraints
{
    public abstract class BaseConstraint
    {
        internal IBackend K = Global.Backend;

        public abstract Tensor Call(Tensor w);
    }
}

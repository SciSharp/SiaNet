using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public abstract class BaseInitializer
    {
        internal IBackend K = Global.CurrentBackend;

        public string Name { get; set; }

        public BaseInitializer(string name)
        {
            Name = name;
        }

        public abstract Tensor Operator(params long[] shape);
    }
}

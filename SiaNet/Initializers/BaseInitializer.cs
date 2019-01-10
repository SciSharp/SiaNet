using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Initializers
{
    public abstract class BaseInitializer : TOps
    {
        public string Name { get; set; }

        public BaseInitializer(string name)
        {
            Name = name;
        }

        public abstract Tensor Operator(params long[] shape);
    }
}

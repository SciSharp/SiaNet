using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace SiaNet.Initializers
{
    public abstract class BaseInitializer
    {
        public string Name { get; set; }

        public BaseInitializer(string name)
        {
            Name = name;
        }

        public abstract Tensor Operator(Tensor tensor);
    }
}

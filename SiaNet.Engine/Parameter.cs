using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Engine;

namespace SiaNet.Engine
{
    public abstract class BaseParameter
    {
        public Tensor Data { get; set; }

        public Tensor Grad { get; set; }

        public string Name { get; set; }

        public BaseParameter(string name, params long[] shape)
        {
            //Name = UUID.GetID(name);
            Name = name;
        }

        public BaseParameter(string name, DataType dataType, params long[] shape)
        {
            //Name = UUID.GetID(name);
            Name = name;
        }
    }
}

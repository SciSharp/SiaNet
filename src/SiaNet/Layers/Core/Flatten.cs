using System;
using System.Collections.Generic;
using System.Text;
using SiaNet.Backend;

namespace SiaNet.Layers
{
    public class Flatten : BaseLayer, ILayer
    {
        public Flatten()
            :base("flatten")
        {

        }

        public Symbol Build(Symbol data)
        {
            return Operators.Flatten(ID, data);
        }
        
    }
}

using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Sigmoid : BaseLayer, ILayer
    {
        public Sigmoid()
            : base("sigmoid")
        {

        }

        public Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "sigmoid")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}

using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class Softplus : BaseLayer, ILayer
    {
        public Softplus()
            : base("softplus")
        {

        }

        public Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "softrelu")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}

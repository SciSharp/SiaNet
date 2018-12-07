using SiaNet.Backend;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Layers.Activations
{
    public class SoftSign : BaseLayer, ILayer
    {
        public SoftSign()
            : base("softsign")
        {

        }

        public Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "softsign")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
